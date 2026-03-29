"""
exp02: Code Stylometry + Gradient Boosting (LightGBM)

Hypothesis: The paper shows TF-IDF SVM (43.05) beats ALL deep models on T1
(DeBERTa: 34.13, ModernBERT: 30.61). This means hand-crafted features that
capture coding STYLE are more robust under distribution shift than learned
token representations.

Key insight from paper:
    - "Variable naming patterns play a central role in distinguishing
       human-written from AI-generated code" (Section D.1)
    - "TF-IDF features are particularly effective in this setting" (Section 5.2.1)
    - AST features HURT when combined with TF-IDF (Section D)

Strategy:
    1. Extract RICH stylometric features (200+ features) focused on
       domain-invariant signals: naming conventions, whitespace patterns,
       comment style, operator preferences, code rhythm
    2. Use CHARACTER-LEVEL n-grams instead of word-level TF-IDF —
       these capture style patterns that transfer across languages
       (e.g., indentation, bracket style, operator spacing)
    3. LightGBM instead of SVM — handles high-dim features better,
       built-in feature importance, faster, and better regularization
    4. Dimensionality reduction with TruncatedSVD (following paper: 500 dims)

Target: Beat SVM TF-IDF (43.05) and approach 50+ on T1.

Usage on Kaggle:
    1. Upload this file to a Kaggle notebook
    2. Run: !pip install datasets lightgbm
    3. Execute: python exp02_stylometric_gbm.py

Author: AICD-Bench Research
"""

import os
import re
import math
import random
import logging
import warnings
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from datasets import load_dataset

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    task: str = "T1"
    num_classes: Dict[str, int] = field(default_factory=lambda: {
        "T1": 2, "T2": 12, "T3": 4,
    })

    # Feature extraction
    char_ngram_range: Tuple[int, int] = (2, 5)  # character n-grams
    word_ngram_range: Tuple[int, int] = (1, 3)  # word n-grams
    max_tfidf_features: int = 10_000  # before SVD
    svd_components: int = 500  # after SVD (following paper)
    max_code_length: int = 10_000  # characters

    # LightGBM
    num_leaves: int = 127
    max_depth: int = -1  # unlimited
    learning_rate: float = 0.05
    n_estimators: int = 2000
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1  # L1
    reg_lambda: float = 1.0  # L2
    early_stopping_rounds: int = 100

    # Data
    max_train_samples: int = 200_000
    max_val_samples: int = 20_000
    max_test_samples: int = 50_000

    # Misc
    seed: int = 42
    n_jobs: int = -1


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# ============================================================================
# Stylometric Feature Extraction (Domain-Invariant Code Fingerprints)
# ============================================================================

class CodeStylometryExtractor:
    """Extract 200+ stylometric features from source code.

    These features capture HOW code is written (authorship style) rather than
    WHAT code does (semantics). This distinction is crucial for OOD robustness:
    - Naming conventions persist across domains/languages
    - Whitespace patterns are author-specific
    - Comment style reveals authorship
    - Code "rhythm" (line length distribution) is consistent
    """

    def extract(self, code: str) -> np.ndarray:
        """Extract all stylometric features from a code snippet."""
        features = []

        # 1. Layout features (whitespace, indentation, line structure)
        features.extend(self._layout_features(code))

        # 2. Naming convention features
        features.extend(self._naming_features(code))

        # 3. Comment style features
        features.extend(self._comment_features(code))

        # 4. Operator and punctuation patterns
        features.extend(self._operator_features(code))

        # 5. Code complexity proxies
        features.extend(self._complexity_features(code))

        # 6. Character distribution features
        features.extend(self._char_distribution_features(code))

        # 7. Keyword usage patterns
        features.extend(self._keyword_features(code))

        # 8. String/literal patterns
        features.extend(self._literal_features(code))

        # 9. Code rhythm features (line length distribution)
        features.extend(self._rhythm_features(code))

        # 10. Structural repetition features
        features.extend(self._repetition_features(code))

        return np.array(features, dtype=np.float32)

    def _layout_features(self, code: str) -> List[float]:
        """Whitespace and indentation patterns — strong authorship signal."""
        lines = code.split("\n")
        non_empty_lines = [l for l in lines if l.strip()]
        num_lines = max(len(lines), 1)
        num_non_empty = max(len(non_empty_lines), 1)

        # Line lengths
        line_lens = [len(l) for l in lines]
        non_empty_lens = [len(l) for l in non_empty_lines] or [0]

        # Indentation
        indents = []
        for line in non_empty_lines:
            indent = len(line) - len(line.lstrip())
            indents.append(indent)
        indents = indents or [0]

        # Tab vs space usage
        tab_lines = sum(1 for l in lines if l.startswith("\t"))
        space_indent_lines = sum(1 for l in lines if re.match(r"^  +\S", l))

        # Trailing whitespace
        trailing_ws = sum(1 for l in lines if l != l.rstrip())

        # Empty line patterns
        empty_lines = sum(1 for l in lines if not l.strip())
        consecutive_empty = sum(1 for i in range(len(lines)-1)
                                if not lines[i].strip() and not lines[i+1].strip())

        # Line ending patterns
        has_final_newline = 1.0 if code.endswith("\n") else 0.0

        return [
            # Line count features
            num_lines,
            num_non_empty,
            empty_lines / num_lines,

            # Line length statistics
            np.mean(non_empty_lens),
            np.median(non_empty_lens),
            np.std(non_empty_lens),
            max(non_empty_lens),
            min(non_empty_lens),
            np.percentile(non_empty_lens, 25),
            np.percentile(non_empty_lens, 75),

            # Indentation statistics
            np.mean(indents),
            np.median(indents),
            np.std(indents),
            max(indents),
            len(set(indents)),  # number of distinct indentation levels

            # Tab vs space
            tab_lines / num_non_empty,
            space_indent_lines / num_non_empty,
            1.0 if tab_lines > space_indent_lines else 0.0,

            # Whitespace hygiene
            trailing_ws / num_lines,
            consecutive_empty / max(empty_lines, 1),
            has_final_newline,
        ]

    def _naming_features(self, code: str) -> List[float]:
        """Variable/function naming conventions — THE key signal per paper."""
        # Extract identifiers (simplified, works across languages)
        identifiers = re.findall(r'\b[a-zA-Z_]\w{1,}\b', code)
        if not identifiers:
            return [0.0] * 20

        total = max(len(identifiers), 1)

        # Naming style counts
        snake_case = sum(1 for i in identifiers if '_' in i and i == i.lower())
        camel_case = sum(1 for i in identifiers
                         if not '_' in i and any(c.isupper() for c in i[1:]) and i[0].islower())
        pascal_case = sum(1 for i in identifiers
                          if not '_' in i and i[0].isupper() and any(c.islower() for c in i))
        upper_case = sum(1 for i in identifiers if i == i.upper() and len(i) > 1 and '_' in i)
        single_char = sum(1 for i in identifiers if len(i) == 1)
        starts_underscore = sum(1 for i in identifiers if i.startswith('_'))
        double_underscore = sum(1 for i in identifiers if i.startswith('__') and i.endswith('__'))

        # Identifier length statistics
        id_lens = [len(i) for i in identifiers]

        # Identifier uniqueness
        unique_ids = len(set(identifiers))

        # Common AI naming patterns (verbose, descriptive)
        long_names = sum(1 for i in identifiers if len(i) > 15)
        short_names = sum(1 for i in identifiers if len(i) <= 3)

        # Prefix patterns (AI models tend to use consistent prefixes)
        prefixes = Counter(i[:3] for i in identifiers if len(i) >= 3)
        prefix_entropy = self._entropy(list(prefixes.values()))

        return [
            snake_case / total,
            camel_case / total,
            pascal_case / total,
            upper_case / total,
            single_char / total,
            starts_underscore / total,
            double_underscore / total,
            long_names / total,
            short_names / total,

            np.mean(id_lens),
            np.median(id_lens),
            np.std(id_lens),
            max(id_lens),

            unique_ids / total,
            unique_ids,
            total,

            prefix_entropy,

            # Ratio features
            snake_case / max(camel_case, 1),
            long_names / max(short_names, 1),
            single_char / max(total - single_char, 1),
        ]

    def _comment_features(self, code: str) -> List[float]:
        """Comment patterns — AI tends to over-comment or under-comment."""
        lines = code.split("\n")
        num_lines = max(len(lines), 1)
        code_len = max(len(code), 1)

        # Comment types
        single_line_hash = sum(1 for l in lines if l.strip().startswith('#'))
        single_line_slash = sum(1 for l in lines if l.strip().startswith('//'))
        inline_hash = sum(1 for l in lines
                          if '#' in l and not l.strip().startswith('#') and '#include' not in l)
        inline_slash = sum(1 for l in lines
                           if '//' in l and not l.strip().startswith('//'))

        # Block comments
        block_comment_starts = code.count('/*')
        docstring_double = code.count('"""')
        docstring_single = code.count("'''")

        # Comment content analysis
        comment_chars = 0
        for l in lines:
            stripped = l.strip()
            if stripped.startswith('#') or stripped.startswith('//'):
                comment_chars += len(stripped)

        # TODO/FIXME/NOTE patterns (AI rarely uses these)
        todo_count = len(re.findall(r'\b(TODO|FIXME|HACK|NOTE|XXX|BUG)\b', code))

        return [
            (single_line_hash + single_line_slash) / num_lines,
            single_line_hash / num_lines,
            single_line_slash / num_lines,
            (inline_hash + inline_slash) / num_lines,
            block_comment_starts / num_lines,
            (docstring_double + docstring_single) / 2.0 / num_lines,
            comment_chars / code_len,
            todo_count / num_lines,
            1.0 if docstring_double > 0 or docstring_single > 0 else 0.0,
        ]

    def _operator_features(self, code: str) -> List[float]:
        """Operator and punctuation usage patterns."""
        code_len = max(len(code), 1)

        # Spacing around operators (AI tends to be consistent)
        spaced_eq = len(re.findall(r' = ', code))
        unspaced_eq = len(re.findall(r'[^ ]=|=[^ =]', code))
        spaced_op = len(re.findall(r' [+\-*/] ', code))

        # Bracket styles
        open_same_line = len(re.findall(r'\)\s*\{', code))  # ) {
        open_next_line = len(re.findall(r'\)\s*\n\s*\{', code))  # )\n{

        # Semicolons
        semicolons = code.count(';')

        # Comma spacing
        spaced_comma = len(re.findall(r', ', code))
        unspaced_comma = len(re.findall(r',[^ \n]', code))

        # Parentheses density
        parens = code.count('(') + code.count(')')
        brackets = code.count('[') + code.count(']')
        braces = code.count('{') + code.count('}')

        # Comparison operators
        double_eq = code.count('==')
        not_eq = code.count('!=')
        triple_eq = code.count('===')

        return [
            spaced_eq / max(spaced_eq + unspaced_eq, 1),
            spaced_op / code_len * 100,
            open_same_line / max(open_same_line + open_next_line, 1),
            semicolons / code_len * 100,
            spaced_comma / max(spaced_comma + unspaced_comma, 1),
            parens / code_len * 100,
            brackets / code_len * 100,
            braces / code_len * 100,
            double_eq / code_len * 100,
            not_eq / code_len * 100,
            triple_eq / code_len * 100,
        ]

    def _complexity_features(self, code: str) -> List[float]:
        """Code complexity proxies."""
        lines = code.split("\n")
        num_lines = max(len(lines), 1)

        # Control flow density
        ifs = len(re.findall(r'\bif\b', code))
        elses = len(re.findall(r'\belse\b', code))
        fors = len(re.findall(r'\bfor\b', code))
        whiles = len(re.findall(r'\bwhile\b', code))
        returns = len(re.findall(r'\breturn\b', code))
        breaks = len(re.findall(r'\bbreak\b', code))
        continues = len(re.findall(r'\bcontinue\b', code))

        # Function/class definitions
        funcs = len(re.findall(r'\b(def|function|func|fn)\s+\w+', code))
        classes = len(re.findall(r'\b(class|struct|interface)\s+\w+', code))

        # Import density
        imports = len(re.findall(r'\b(import|include|require|using)\b', code))

        # Exception handling
        try_catch = len(re.findall(r'\b(try|catch|except|finally)\b', code))

        # Nesting depth (approximate)
        max_depth = 0
        current_depth = 0
        for char in code:
            if char in '({[':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in ')}]':
                current_depth = max(0, current_depth - 1)

        return [
            ifs / num_lines,
            (ifs + elses) / num_lines,
            (fors + whiles) / num_lines,
            returns / num_lines,
            (breaks + continues) / num_lines,
            funcs / num_lines,
            classes / num_lines,
            imports / num_lines,
            try_catch / num_lines,
            max_depth,
            elses / max(ifs, 1),  # else/if ratio
            returns / max(funcs, 1),  # returns per function
        ]

    def _char_distribution_features(self, code: str) -> List[float]:
        """Character distribution — captures language and style."""
        code_len = max(len(code), 1)

        alpha = sum(c.isalpha() for c in code) / code_len
        digit = sum(c.isdigit() for c in code) / code_len
        space = sum(c == ' ' for c in code) / code_len
        tab = sum(c == '\t' for c in code) / code_len
        newline = sum(c == '\n' for c in code) / code_len
        underscore = code.count('_') / code_len
        dot = code.count('.') / code_len
        colon = code.count(':') / code_len
        at_sign = code.count('@') / code_len  # decorators

        # Uppercase ratio (among alphabetic)
        alpha_chars = [c for c in code if c.isalpha()]
        upper_ratio = sum(c.isupper() for c in alpha_chars) / max(len(alpha_chars), 1)

        # Unique character count
        unique_chars = len(set(code))

        # Character entropy
        char_counts = Counter(code)
        char_entropy = self._entropy(list(char_counts.values()))

        return [
            alpha, digit, space, tab, newline,
            underscore, dot, colon, at_sign,
            upper_ratio, unique_chars / code_len, char_entropy,
        ]

    def _keyword_features(self, code: str) -> List[float]:
        """Programming keyword usage patterns."""
        # Language-agnostic keywords
        keywords = {
            'if': r'\bif\b', 'else': r'\belse\b', 'for': r'\bfor\b',
            'while': r'\bwhile\b', 'return': r'\breturn\b', 'def': r'\bdef\b',
            'class': r'\bclass\b', 'import': r'\bimport\b', 'from': r'\bfrom\b',
            'try': r'\btry\b', 'except': r'\bexcept\b', 'catch': r'\bcatch\b',
            'throw': r'\bthrow\b', 'raise': r'\braise\b',
            'new': r'\bnew\b', 'this': r'\bthis\b', 'self': r'\bself\b',
            'null': r'\b(null|None|nil|nullptr)\b',
            'true': r'\b(true|True|TRUE)\b', 'false': r'\b(false|False|FALSE)\b',
            'void': r'\bvoid\b', 'int': r'\bint\b', 'string': r'\bstring\b',
            'public': r'\bpublic\b', 'private': r'\bprivate\b',
            'static': r'\bstatic\b', 'const': r'\bconst\b',
            'async': r'\basync\b', 'await': r'\bawait\b',
            'lambda': r'\blambda\b', 'yield': r'\byield\b',
        }

        code_len = max(len(code), 1)
        features = []
        for name, pattern in keywords.items():
            count = len(re.findall(pattern, code))
            features.append(count / code_len * 1000)  # per 1K chars

        return features

    def _literal_features(self, code: str) -> List[float]:
        """String and numeric literal patterns."""
        code_len = max(len(code), 1)

        # String literals
        double_strings = len(re.findall(r'"[^"]*"', code))
        single_strings = len(re.findall(r"'[^']*'", code))
        f_strings = len(re.findall(r'f"[^"]*"', code)) + len(re.findall(r"f'[^']*'", code))
        template_literals = code.count('`')

        # Numeric literals
        integers = len(re.findall(r'\b\d+\b', code))
        floats = len(re.findall(r'\b\d+\.\d+\b', code))
        hex_literals = len(re.findall(r'0x[0-9a-fA-F]+', code))

        # Magic numbers (non-0, non-1 bare integers)
        magic_numbers = len(re.findall(r'(?<![a-zA-Z_])[2-9]\d*(?![a-zA-Z_\.])', code))

        return [
            double_strings / code_len * 100,
            single_strings / code_len * 100,
            double_strings / max(single_strings, 1),  # quote preference
            f_strings / code_len * 100,
            template_literals / code_len * 100,
            integers / code_len * 100,
            floats / code_len * 100,
            hex_literals / code_len * 100,
            magic_numbers / code_len * 100,
        ]

    def _rhythm_features(self, code: str) -> List[float]:
        """Code rhythm — distribution of line lengths (author-specific pattern)."""
        lines = code.split("\n")
        non_empty = [l for l in lines if l.strip()]
        if not non_empty:
            return [0.0] * 10

        lens = [len(l) for l in non_empty]

        # Line length histogram (10 bins)
        bins = [0, 20, 40, 60, 80, 100, 120, 160, 200, 300, float('inf')]
        hist = np.histogram(lens, bins=bins)[0]
        hist = hist / max(len(non_empty), 1)

        return hist.tolist()

    def _repetition_features(self, code: str) -> List[float]:
        """Structural repetition patterns — AI code tends to be more repetitive."""
        lines = code.split("\n")
        stripped_lines = [l.strip() for l in lines if l.strip()]
        if not stripped_lines:
            return [0.0] * 5

        total = len(stripped_lines)

        # Duplicate lines
        line_counts = Counter(stripped_lines)
        duplicate_lines = sum(1 for c in line_counts.values() if c > 1)
        max_line_repeat = max(line_counts.values())

        # Duplicate line ratio
        unique_lines = len(line_counts)

        # Bigram repetition (consecutive line pairs)
        bigrams = [(stripped_lines[i], stripped_lines[i+1]) for i in range(len(stripped_lines)-1)]
        bigram_counts = Counter(bigrams)
        duplicate_bigrams = sum(1 for c in bigram_counts.values() if c > 1)

        # Line entropy (how varied are the lines)
        line_entropy = self._entropy(list(line_counts.values()))

        return [
            duplicate_lines / total,
            max_line_repeat / total,
            unique_lines / total,
            duplicate_bigrams / max(len(bigrams), 1),
            line_entropy,
        ]

    @staticmethod
    def _entropy(counts: List[int]) -> float:
        """Shannon entropy of a distribution."""
        if not counts:
            return 0.0
        total = sum(counts)
        if total == 0:
            return 0.0
        probs = [c / total for c in counts if c > 0]
        return -sum(p * math.log2(p) for p in probs)


# ============================================================================
# Feature Pipeline
# ============================================================================

class FeaturePipeline:
    """Combine TF-IDF (char + word n-grams) with stylometric features."""

    def __init__(self, config: Config):
        self.config = config
        self.stylometry = CodeStylometryExtractor()

        # Character-level TF-IDF (captures style across languages)
        self.char_tfidf = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=config.char_ngram_range,
            max_features=config.max_tfidf_features,
            sublinear_tf=True,
            strip_accents='unicode',
        )

        # Word-level TF-IDF (captures keyword patterns)
        self.word_tfidf = TfidfVectorizer(
            analyzer='word',
            ngram_range=config.word_ngram_range,
            max_features=config.max_tfidf_features,
            sublinear_tf=True,
            token_pattern=r'(?u)\b\w+\b',  # include single-char tokens
        )

        self.char_svd = TruncatedSVD(n_components=config.svd_components, random_state=config.seed)
        self.word_svd = TruncatedSVD(n_components=config.svd_components, random_state=config.seed)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_transform(self, codes: List[str]) -> np.ndarray:
        """Fit pipeline and transform training data."""
        logger.info("Extracting stylometric features...")
        stylo_features = np.array([self.stylometry.extract(c[:self.config.max_code_length])
                                    for c in codes])
        logger.info(f"  Stylometric features shape: {stylo_features.shape}")

        logger.info("Fitting character TF-IDF...")
        char_tfidf = self.char_tfidf.fit_transform(codes)
        logger.info(f"  Char TF-IDF shape: {char_tfidf.shape}")

        logger.info("Fitting word TF-IDF...")
        word_tfidf = self.word_tfidf.fit_transform(codes)
        logger.info(f"  Word TF-IDF shape: {word_tfidf.shape}")

        logger.info("Applying SVD...")
        char_svd = self.char_svd.fit_transform(char_tfidf)
        word_svd = self.word_svd.fit_transform(word_tfidf)
        logger.info(f"  Char SVD explained variance: {self.char_svd.explained_variance_ratio_.sum():.3f}")
        logger.info(f"  Word SVD explained variance: {self.word_svd.explained_variance_ratio_.sum():.3f}")

        # Combine all features
        combined = np.hstack([stylo_features, char_svd, word_svd])
        logger.info(f"  Combined feature shape: {combined.shape}")

        # Scale
        combined = self.scaler.fit_transform(combined)
        self.is_fitted = True

        return combined

    def transform(self, codes: List[str]) -> np.ndarray:
        """Transform new data using fitted pipeline."""
        assert self.is_fitted, "Pipeline not fitted. Call fit_transform first."

        stylo_features = np.array([self.stylometry.extract(c[:self.config.max_code_length])
                                    for c in codes])

        char_tfidf = self.char_tfidf.transform(codes)
        word_tfidf = self.word_tfidf.transform(codes)

        char_svd = self.char_svd.transform(char_tfidf)
        word_svd = self.word_svd.transform(word_tfidf)

        combined = np.hstack([stylo_features, char_svd, word_svd])
        combined = self.scaler.transform(combined)

        return combined


# ============================================================================
# LightGBM Trainer
# ============================================================================

class GBMTrainer:
    """LightGBM trainer with early stopping."""

    def __init__(self, config: Config, num_classes: int):
        self.config = config
        self.num_classes = num_classes
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        """Train LightGBM with early stopping on validation set."""
        params = {
            'objective': 'binary' if self.num_classes == 2 else 'multiclass',
            'metric': 'binary_logloss' if self.num_classes == 2 else 'multi_logloss',
            'num_leaves': self.config.num_leaves,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'min_child_samples': self.config.min_child_samples,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'seed': self.config.seed,
            'verbose': -1,
            'n_jobs': self.config.n_jobs,
            'is_unbalance': True,
        }
        if self.num_classes > 2:
            params['num_class'] = self.num_classes

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds),
        ]

        logger.info("Training LightGBM...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        logger.info(f"Best iteration: {self.model.best_iteration}")
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        raw_preds = self.model.predict(X)
        if self.num_classes == 2:
            return (raw_preds > 0.5).astype(int)
        else:
            return raw_preds.argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        raw_preds = self.model.predict(X)
        if self.num_classes == 2:
            return np.column_stack([1 - raw_preds, raw_preds])
        return raw_preds

    def feature_importance(self, feature_names: List[str] = None, top_k: int = 30):
        """Log top feature importances."""
        importance = self.model.feature_importance(importance_type='gain')
        if feature_names is None:
            feature_names = [f"f_{i}" for i in range(len(importance))]

        sorted_idx = np.argsort(importance)[::-1][:top_k]
        logger.info(f"\nTop {top_k} feature importances (gain):")
        for i, idx in enumerate(sorted_idx):
            logger.info(f"  {i+1:3d}. {feature_names[idx]:40s} = {importance[idx]:.2f}")


# ============================================================================
# Main
# ============================================================================

def main(task: str = "T1", config: Optional[Config] = None):
    if config is None:
        config = Config(task=task)
    set_seed(config.seed)

    if not HAS_LIGHTGBM:
        logger.error("LightGBM not installed. Run: pip install lightgbm")
        return

    logger.info("=" * 60)
    logger.info(f"exp02 - Code Stylometry + LightGBM")
    logger.info(f"Task: {config.task}")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading AICD-Bench task {config.task}...")
    ds = load_dataset("AICD-bench/AICD-Bench", name=config.task)

    train_data = ds["train"]
    val_data = ds["validation"]
    test_data = ds["test"]

    # Subsample
    if len(train_data) > config.max_train_samples:
        indices = random.sample(range(len(train_data)), config.max_train_samples)
        train_data = train_data.select(indices)
    if len(val_data) > config.max_val_samples:
        indices = random.sample(range(len(val_data)), config.max_val_samples)
        val_data = val_data.select(indices)
    if len(test_data) > config.max_test_samples:
        indices = random.sample(range(len(test_data)), config.max_test_samples)
        test_data = test_data.select(indices)

    logger.info(f"Data sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    num_classes = len(set(train_data["label"]))
    logger.info(f"Number of classes: {num_classes}")

    # Extract features
    pipeline = FeaturePipeline(config)

    train_codes = train_data["code"]
    val_codes = val_data["code"]
    test_codes = test_data["code"]

    X_train = pipeline.fit_transform(train_codes)
    X_val = pipeline.transform(val_codes)
    X_test = pipeline.transform(test_codes)

    y_train = np.array(train_data["label"])
    y_val = np.array(val_data["label"])
    y_test = np.array(test_data["label"])

    # Train
    trainer = GBMTrainer(config, num_classes)
    trainer.train(X_train, y_train, X_val, y_val)

    # Feature importance
    n_stylo = len(CodeStylometryExtractor().extract("dummy code"))
    feature_names = (
        [f"stylo_{i}" for i in range(n_stylo)] +
        [f"char_svd_{i}" for i in range(config.svd_components)] +
        [f"word_svd_{i}" for i in range(config.svd_components)]
    )
    trainer.feature_importance(feature_names)

    # Evaluate on validation
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION EVALUATION")
    logger.info("=" * 60)
    val_preds = trainer.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    logger.info(f"Val Macro-F1: {val_f1:.4f}")
    logger.info(f"\n{classification_report(y_val, val_preds, digits=4)}")

    # Evaluate on test
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST EVALUATION")
    logger.info("=" * 60)
    test_preds = trainer.predict(X_test)
    test_f1 = f1_score(y_test, test_preds, average="macro")
    logger.info(f"Test Macro-F1: {test_f1:.4f}")
    logger.info(f"\n{classification_report(y_test, test_preds, digits=4)}")

    logger.info(f"\n*** Final Test Macro-F1: {test_f1:.4f} ***")

    # Save model
    os.makedirs(config.save_dir if hasattr(config, 'save_dir') else "./exp02_checkpoints", exist_ok=True)
    save_path = os.path.join("./exp02_checkpoints", f"exp02_{config.task}_model.txt")
    trainer.model.save_model(save_path)
    logger.info(f"Saved model to {save_path}")

    return test_f1


if __name__ == "__main__":
    main(task="T1")
