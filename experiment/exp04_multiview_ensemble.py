"""
exp04: Multi-View Stacking Ensemble

Hypothesis: No single model captures all the signals needed for OOD robustness.
Neural models capture semantic patterns but overfit to domain.
Statistical models capture style patterns but miss semantics.
Combining DIVERSE views via stacking ensemble maximizes OOD robustness.

Key insight from paper:
    - Gemini CoT (62.31) >> best trained model (43.05 SVM TF-IDF)
    - Deep models and statistical models fail in DIFFERENT ways
    - "Variable naming patterns" + "TF-IDF features" are complementary

Strategy:
    1. View 1: Character n-gram TF-IDF + LightGBM (captures style/naming)
    2. View 2: Word n-gram TF-IDF + LightGBM (captures keyword patterns)
    3. View 3: Stylometric features + LightGBM (captures code structure)
    4. View 4: ModernBERT frozen embeddings + MLP (captures semantics)
    5. Meta-learner: Logistic Regression stacking on held-out predictions
       from all 4 views — learns optimal weighting per-sample

Why stacking works for OOD:
    - Different views fail on different OOD samples
    - Meta-learner learns to trust the most robust view per input
    - Diversity reduces correlation of errors across views

Target: Beat Gemini CoT (62.31) on T1, beat ModernBERT (32.84) on T2,
        beat ModernBERT (61.65) on T3.

Usage on Kaggle:
    1. Upload this file to a Kaggle notebook
    2. Run: !pip install datasets transformers lightgbm
    3. Execute: python exp04_multiview_ensemble.py

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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
try:
    from torch.amp import autocast as _autocast, GradScaler
    _NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast as _autocast, GradScaler
    _NEW_AMP = False

def autocast(device_type="cuda", enabled=True):
    if _NEW_AMP:
        return _autocast(device_type=device_type, enabled=enabled)
    else:
        return _autocast(enabled=enabled)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from datasets import load_dataset

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from transformers import AutoTokenizer, AutoModel, AutoConfig

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

    # Encoder for View 4
    encoder_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512  # keep short for embedding extraction speed
    embedding_batch_size: int = 64

    # TF-IDF settings
    char_ngram_range: Tuple[int, int] = (2, 5)
    word_ngram_range: Tuple[int, int] = (1, 3)
    max_tfidf_features: int = 10_000
    svd_components: int = 300

    # LightGBM shared settings
    lgb_num_leaves: int = 127
    lgb_learning_rate: float = 0.05
    lgb_n_estimators: int = 1500
    lgb_min_child_samples: int = 50
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8
    lgb_reg_alpha: float = 0.1
    lgb_reg_lambda: float = 1.0
    lgb_early_stopping: int = 100

    # Stacking
    stacking_cv_folds: int = 5  # K-fold for generating meta-features
    meta_C: float = 1.0  # Logistic Regression regularization

    # Data
    max_train_samples: int = 200_000
    max_val_samples: int = 20_000
    max_test_samples: int = 50_000
    num_workers: int = 2

    # Misc
    seed: int = 42
    fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_jobs: int = -1


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Stylometric Features (reused from exp02, simplified)
# ============================================================================

def extract_stylometric_features(code: str) -> np.ndarray:
    """Extract ~100 stylometric features from code."""
    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]
    num_lines = max(len(lines), 1)
    num_non_empty = max(len(non_empty), 1)
    code_len = max(len(code), 1)

    features = []

    # --- Layout (15) ---
    line_lens = [len(l) for l in non_empty] or [0]
    indents = [len(l) - len(l.lstrip()) for l in non_empty] or [0]
    empty_lines = sum(1 for l in lines if not l.strip())
    tab_lines = sum(1 for l in lines if l.startswith("\t"))
    space_lines = sum(1 for l in lines if re.match(r"^  +\S", l))
    trailing_ws = sum(1 for l in lines if l != l.rstrip())

    features.extend([
        num_lines, num_non_empty, empty_lines / num_lines,
        np.mean(line_lens), np.std(line_lens), max(line_lens), min(line_lens),
        np.mean(indents), np.std(indents), max(indents), len(set(indents)),
        tab_lines / num_non_empty, space_lines / num_non_empty,
        trailing_ws / num_lines,
        1.0 if code.endswith("\n") else 0.0,
    ])

    # --- Naming (12) ---
    identifiers = re.findall(r'\b[a-zA-Z_]\w{1,}\b', code)
    total_ids = max(len(identifiers), 1)
    snake = sum(1 for i in identifiers if '_' in i and i == i.lower())
    camel = sum(1 for i in identifiers if '_' not in i and any(c.isupper() for c in i[1:]) and i[0].islower())
    single = sum(1 for i in identifiers if len(i) == 1)
    long = sum(1 for i in identifiers if len(i) > 15)
    id_lens = [len(i) for i in identifiers] or [0]

    features.extend([
        snake / total_ids, camel / total_ids, single / total_ids, long / total_ids,
        np.mean(id_lens), np.std(id_lens), max(id_lens),
        len(set(identifiers)) / total_ids,
        snake / max(camel, 1), single / max(total_ids - single, 1),
        total_ids / num_lines,
        sum(1 for i in identifiers if i.startswith('_')) / total_ids,
    ])

    # --- Comments (6) ---
    hash_comments = sum(1 for l in lines if l.strip().startswith('#'))
    slash_comments = sum(1 for l in lines if l.strip().startswith('//'))
    block_comments = code.count('/*')
    docstrings = code.count('"""') + code.count("'''")
    todo_count = len(re.findall(r'\b(TODO|FIXME|HACK|NOTE)\b', code))

    features.extend([
        (hash_comments + slash_comments) / num_lines,
        hash_comments / num_lines, slash_comments / num_lines,
        block_comments / num_lines, docstrings / num_lines,
        todo_count / num_lines,
    ])

    # --- Operators (8) ---
    spaced_eq = len(re.findall(r' = ', code))
    unspaced_eq = len(re.findall(r'[^ ]=|=[^ =]', code))
    parens = (code.count('(') + code.count(')')) / code_len * 100
    brackets = (code.count('[') + code.count(']')) / code_len * 100
    braces = (code.count('{') + code.count('}')) / code_len * 100
    semicolons = code.count(';') / code_len * 100

    features.extend([
        spaced_eq / max(spaced_eq + unspaced_eq, 1),
        parens, brackets, braces, semicolons,
        code.count('==') / code_len * 100,
        code.count('!=') / code_len * 100,
        code.count('===') / code_len * 100,
    ])

    # --- Complexity (10) ---
    features.extend([
        len(re.findall(r'\bif\b', code)) / num_lines,
        len(re.findall(r'\b(for|while)\b', code)) / num_lines,
        len(re.findall(r'\breturn\b', code)) / num_lines,
        len(re.findall(r'\b(def|function|func|fn)\s+\w+', code)) / num_lines,
        len(re.findall(r'\b(class|struct|interface)\s+\w+', code)) / num_lines,
        len(re.findall(r'\b(import|include|require|using)\b', code)) / num_lines,
        len(re.findall(r'\b(try|catch|except)\b', code)) / num_lines,
        len(re.findall(r'\b(break|continue)\b', code)) / num_lines,
        len(re.findall(r'\belse\b', code)) / max(len(re.findall(r'\bif\b', code)), 1),
        len(re.findall(r'\breturn\b', code)) / max(len(re.findall(r'\b(def|function|func|fn)\s+\w+', code)), 1),
    ])

    # --- Character distribution (10) ---
    alpha = sum(c.isalpha() for c in code) / code_len
    digit = sum(c.isdigit() for c in code) / code_len
    space = sum(c == ' ' for c in code) / code_len
    underscore = code.count('_') / code_len
    dot = code.count('.') / code_len

    alpha_chars = [c for c in code if c.isalpha()]
    upper_ratio = sum(c.isupper() for c in alpha_chars) / max(len(alpha_chars), 1)

    features.extend([
        alpha, digit, space, underscore, dot,
        code.count('\t') / code_len, code.count('\n') / code_len,
        code.count(':') / code_len, code.count('@') / code_len,
        upper_ratio,
    ])

    # --- Rhythm (10 bins) ---
    bins = [0, 20, 40, 60, 80, 100, 120, 160, 200, 300, float('inf')]
    hist = np.histogram(line_lens, bins=bins)[0].astype(float)
    hist = hist / max(num_non_empty, 1)
    features.extend(hist.tolist())

    # --- Repetition (5) ---
    from collections import Counter
    stripped_lines = [l.strip() for l in non_empty]
    line_counts = Counter(stripped_lines)
    total_stripped = max(len(stripped_lines), 1)

    dup_lines = sum(1 for c in line_counts.values() if c > 1)
    unique_lines = len(line_counts)

    features.extend([
        dup_lines / total_stripped,
        max(line_counts.values()) / total_stripped if line_counts else 0,
        unique_lines / total_stripped,
        len([c for c in line_counts.values() if c > 2]) / total_stripped,
        # Line entropy
        _entropy(list(line_counts.values())),
    ])

    # --- Keyword density (20 keywords) ---
    keywords = [
        r'\bif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b', r'\breturn\b',
        r'\bdef\b', r'\bclass\b', r'\bimport\b', r'\btry\b', r'\bexcept\b',
        r'\bnew\b', r'\bthis\b', r'\bself\b', r'\b(null|None|nil)\b',
        r'\bvoid\b', r'\bpublic\b', r'\bstatic\b', r'\bconst\b',
        r'\basync\b', r'\blambda\b',
    ]
    for kw in keywords:
        features.append(len(re.findall(kw, code)) / code_len * 1000)

    return np.array(features, dtype=np.float32)


def _entropy(counts):
    if not counts:
        return 0.0
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


# ============================================================================
# View 1: Character n-gram TF-IDF + LightGBM
# ============================================================================

class CharNgramView:
    """Character n-grams capture style patterns across languages."""

    def __init__(self, config: Config):
        self.config = config
        self.tfidf = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=config.char_ngram_range,
            max_features=config.max_tfidf_features,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=config.svd_components, random_state=config.seed)
        self.scaler = StandardScaler()

    def fit_transform(self, codes):
        X = self.tfidf.fit_transform(codes)
        X = self.svd.fit_transform(X)
        X = self.scaler.fit_transform(X)
        return X

    def transform(self, codes):
        X = self.tfidf.transform(codes)
        X = self.svd.transform(X)
        X = self.scaler.transform(X)
        return X


# ============================================================================
# View 2: Word n-gram TF-IDF + LightGBM
# ============================================================================

class WordNgramView:
    """Word n-grams capture keyword and identifier patterns."""

    def __init__(self, config: Config):
        self.config = config
        self.tfidf = TfidfVectorizer(
            analyzer='word',
            ngram_range=config.word_ngram_range,
            max_features=config.max_tfidf_features,
            sublinear_tf=True,
            token_pattern=r'(?u)\b\w+\b',
        )
        self.svd = TruncatedSVD(n_components=config.svd_components, random_state=config.seed)
        self.scaler = StandardScaler()

    def fit_transform(self, codes):
        X = self.tfidf.fit_transform(codes)
        X = self.svd.fit_transform(X)
        X = self.scaler.fit_transform(X)
        return X

    def transform(self, codes):
        X = self.tfidf.transform(codes)
        X = self.svd.transform(X)
        X = self.scaler.transform(X)
        return X


# ============================================================================
# View 3: Stylometric Features + LightGBM
# ============================================================================

class StylometryView:
    """Hand-crafted code style features."""

    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()

    def fit_transform(self, codes):
        X = np.array([extract_stylometric_features(c[:10000]) for c in codes])
        # Replace NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.fit_transform(X)
        return X

    def transform(self, codes):
        X = np.array([extract_stylometric_features(c[:10000]) for c in codes])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(X)
        return X


# ============================================================================
# View 4: Frozen ModernBERT Embeddings + LightGBM
# ============================================================================

class EmbeddingView:
    """Frozen transformer embeddings — capture semantic patterns."""

    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.scaler = StandardScaler()

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading {self.config.encoder_name} for embedding extraction...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.encoder_name)
            self.model = AutoModel.from_pretrained(self.config.encoder_name)
            self.model.eval()
            if self.config.device == "cuda":
                self.model = self.model.cuda()

    @torch.no_grad()
    def _extract_embeddings(self, codes: List[str]) -> np.ndarray:
        self._load_model()

        all_embeddings = []
        batch_size = self.config.embedding_batch_size

        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i+batch_size]

            encoding = self.tokenizer(
                batch_codes,
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            if self.config.device == "cuda":
                encoding = {k: v.cuda() for k, v in encoding.items()}

            with autocast(device_type=self.config.device, enabled=self.config.fp16):
                outputs = self.model(**encoding, output_hidden_states=True)

            # Pool: [CLS] from last layer + mean from last 4 layers
            last_hidden = outputs.hidden_states[-1]
            cls_emb = last_hidden[:, 0]  # (B, hidden)

            mask = encoding["attention_mask"].unsqueeze(-1).float()
            # Mean of last 4 layers
            multi_layer = torch.stack(outputs.hidden_states[-4:]).mean(dim=0)
            mean_emb = (multi_layer * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            emb = torch.cat([cls_emb, mean_emb], dim=-1)  # (B, hidden*2)
            all_embeddings.append(emb.cpu().float().numpy())

            if (i // batch_size + 1) % 50 == 0:
                logger.info(f"  Extracted {min(i+batch_size, len(codes))}/{len(codes)} embeddings")

        return np.vstack(all_embeddings)

    def fit_transform(self, codes):
        X = self._extract_embeddings(codes)
        X = self.scaler.fit_transform(X)
        return X

    def transform(self, codes):
        X = self._extract_embeddings(codes)
        X = self.scaler.transform(X)
        return X

    def cleanup(self):
        """Free GPU memory after embedding extraction."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============================================================================
# LightGBM Base Learner
# ============================================================================

def train_lgb(X_train, y_train, X_val, y_val, config: Config,
              num_classes: int, view_name: str) -> lgb.Booster:
    """Train a LightGBM model for one view."""
    params = {
        'objective': 'binary' if num_classes == 2 else 'multiclass',
        'metric': 'binary_logloss' if num_classes == 2 else 'multi_logloss',
        'num_leaves': config.lgb_num_leaves,
        'learning_rate': config.lgb_learning_rate,
        'min_child_samples': config.lgb_min_child_samples,
        'subsample': config.lgb_subsample,
        'colsample_bytree': config.lgb_colsample_bytree,
        'reg_alpha': config.lgb_reg_alpha,
        'reg_lambda': config.lgb_reg_lambda,
        'seed': config.seed,
        'verbose': -1,
        'n_jobs': config.n_jobs,
        'is_unbalance': True,
    }
    if num_classes > 2:
        params['num_class'] = num_classes

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    callbacks = [
        lgb.log_evaluation(period=200),
        lgb.early_stopping(stopping_rounds=config.lgb_early_stopping),
    ]

    logger.info(f"Training LightGBM for {view_name}...")
    model = lgb.train(
        params, train_data,
        num_boost_round=config.lgb_n_estimators,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    logger.info(f"  {view_name} best iteration: {model.best_iteration}")
    return model


def lgb_predict_proba(model, X, num_classes):
    """Get probability predictions from LightGBM."""
    raw = model.predict(X)
    if num_classes == 2:
        return np.column_stack([1 - raw, raw])
    return raw


# ============================================================================
# Stacking Ensemble
# ============================================================================

class StackingEnsemble:
    """Multi-view stacking ensemble with K-fold meta-feature generation.

    Level 0: Train separate LightGBM models on each view's features
    Level 1: Train Logistic Regression on concatenated probability predictions

    K-fold cross-validation ensures meta-features don't leak:
    - Split training data into K folds
    - For each fold: train on K-1 folds, predict on held-out fold
    - Concatenate held-out predictions as meta-features
    """

    def __init__(self, config: Config, num_classes: int):
        self.config = config
        self.num_classes = num_classes
        self.view_models = {}  # view_name -> list of K lgb models
        self.meta_model = None

    def fit(self, views: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
            y_train: np.ndarray, y_val: np.ndarray):
        """
        Args:
            views: dict of view_name -> (X_train, X_val, X_test)
            y_train: training labels
            y_val: validation labels
        """
        n_train = len(y_train)
        n_classes = self.num_classes
        k_folds = self.config.stacking_cv_folds

        # Generate meta-features via K-fold cross-validation
        meta_train = np.zeros((n_train, 0))
        meta_val = {name: None for name in views}

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.config.seed)

        for view_name, (X_train, X_val, X_test) in views.items():
            logger.info(f"\n{'='*40}")
            logger.info(f"Generating meta-features for {view_name}")
            logger.info(f"{'='*40}")

            view_meta_train = np.zeros((n_train, n_classes))
            fold_models = []

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                logger.info(f"  Fold {fold_idx+1}/{k_folds}")

                X_fold_train = X_train[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train[val_idx]
                y_fold_val = y_train[val_idx]

                model = train_lgb(
                    X_fold_train, y_fold_train,
                    X_fold_val, y_fold_val,
                    self.config, n_classes,
                    f"{view_name}_fold{fold_idx}",
                )

                # Out-of-fold predictions (meta-features)
                view_meta_train[val_idx] = lgb_predict_proba(model, X_fold_val, n_classes)
                fold_models.append(model)

            self.view_models[view_name] = fold_models

            # Average fold models for val/test prediction
            val_probs = np.mean([
                lgb_predict_proba(m, X_val, n_classes) for m in fold_models
            ], axis=0)
            meta_val[view_name] = val_probs

            meta_train = np.hstack([meta_train, view_meta_train])

            # Log per-view performance
            view_val_preds = val_probs.argmax(axis=1)
            view_f1 = f1_score(y_val, view_val_preds, average="macro")
            logger.info(f"  {view_name} Val Macro-F1: {view_f1:.4f}")

        # Stack val meta-features
        meta_val_combined = np.hstack([meta_val[name] for name in views])

        # Train meta-learner
        logger.info(f"\n{'='*40}")
        logger.info("Training meta-learner (Logistic Regression)")
        logger.info(f"Meta-feature shape: {meta_train.shape}")
        logger.info(f"{'='*40}")

        self.meta_model = LogisticRegression(
            C=self.config.meta_C,
            max_iter=1000,
            solver='lbfgs',
            multi_class='multinomial' if n_classes > 2 else 'auto',
            random_state=self.config.seed,
        )
        self.meta_model.fit(meta_train, y_train)

        # Validate meta-learner
        meta_val_preds = self.meta_model.predict(meta_val_combined)
        meta_val_f1 = f1_score(y_val, meta_val_preds, average="macro")
        logger.info(f"Stacked Ensemble Val Macro-F1: {meta_val_f1:.4f}")

        return meta_val_f1

    def predict(self, views: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict using the stacking ensemble.

        Args:
            views: dict of view_name -> X_test features
        """
        n_classes = self.num_classes
        meta_test = np.hstack([
            np.mean([
                lgb_predict_proba(m, views[name], n_classes)
                for m in self.view_models[name]
            ], axis=0)
            for name in self.view_models
        ])

        return self.meta_model.predict(meta_test)

    def predict_proba(self, views: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict probabilities."""
        n_classes = self.num_classes
        meta_test = np.hstack([
            np.mean([
                lgb_predict_proba(m, views[name], n_classes)
                for m in self.view_models[name]
            ], axis=0)
            for name in self.view_models
        ])

        return self.meta_model.predict_proba(meta_test)


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
    logger.info(f"exp04 - Multi-View Stacking Ensemble")
    logger.info(f"Task: {config.task}")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading AICD-Bench task {config.task}...")
    ds = load_dataset("AICD-bench/AICD-Bench", name=config.task)

    train_data = ds["train"]
    val_data = ds["validation"]
    test_data = ds["test"]

    if len(train_data) > config.max_train_samples:
        indices = random.sample(range(len(train_data)), config.max_train_samples)
        train_data = train_data.select(indices)
    if len(val_data) > config.max_val_samples:
        indices = random.sample(range(len(val_data)), config.max_val_samples)
        val_data = val_data.select(indices)
    if len(test_data) > config.max_test_samples:
        indices = random.sample(range(len(test_data)), config.max_test_samples)
        test_data = test_data.select(indices)

    logger.info(f"Data: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    num_classes = len(set(train_data["label"]))
    logger.info(f"Number of classes: {num_classes}")

    train_codes = train_data["code"]
    val_codes = val_data["code"]
    test_codes = test_data["code"]
    y_train = np.array(train_data["label"])
    y_val = np.array(val_data["label"])
    y_test = np.array(test_data["label"])

    # ---- Extract features for all views ----

    # View 1: Character n-grams
    logger.info("\n--- View 1: Character n-grams ---")
    char_view = CharNgramView(config)
    X_train_char = char_view.fit_transform(train_codes)
    X_val_char = char_view.transform(val_codes)
    X_test_char = char_view.transform(test_codes)
    logger.info(f"  Shape: {X_train_char.shape}")

    # View 2: Word n-grams
    logger.info("\n--- View 2: Word n-grams ---")
    word_view = WordNgramView(config)
    X_train_word = word_view.fit_transform(train_codes)
    X_val_word = word_view.transform(val_codes)
    X_test_word = word_view.transform(test_codes)
    logger.info(f"  Shape: {X_train_word.shape}")

    # View 3: Stylometric features
    logger.info("\n--- View 3: Stylometric features ---")
    stylo_view = StylometryView(config)
    X_train_stylo = stylo_view.fit_transform(train_codes)
    X_val_stylo = stylo_view.transform(val_codes)
    X_test_stylo = stylo_view.transform(test_codes)
    logger.info(f"  Shape: {X_train_stylo.shape}")

    # View 4: ModernBERT embeddings
    logger.info("\n--- View 4: ModernBERT frozen embeddings ---")
    emb_view = EmbeddingView(config)
    X_train_emb = emb_view.fit_transform(train_codes)
    X_val_emb = emb_view.transform(val_codes)
    X_test_emb = emb_view.transform(test_codes)
    logger.info(f"  Shape: {X_train_emb.shape}")
    emb_view.cleanup()  # free GPU memory

    # ---- Build ensemble ----
    views = {
        "char_ngram": (X_train_char, X_val_char, X_test_char),
        "word_ngram": (X_train_word, X_val_word, X_test_word),
        "stylometry": (X_train_stylo, X_val_stylo, X_test_stylo),
        "modernbert_emb": (X_train_emb, X_val_emb, X_test_emb),
    }

    ensemble = StackingEnsemble(config, num_classes)
    val_f1 = ensemble.fit(views, y_train, y_val)

    # ---- Final test evaluation ----
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST EVALUATION")
    logger.info("=" * 60)

    test_views = {
        "char_ngram": X_test_char,
        "word_ngram": X_test_word,
        "stylometry": X_test_stylo,
        "modernbert_emb": X_test_emb,
    }

    # Individual view test performance
    for view_name in views:
        _, _, X_test_view = views[view_name]
        view_preds = np.mean([
            lgb_predict_proba(m, X_test_view, num_classes)
            for m in ensemble.view_models[view_name]
        ], axis=0).argmax(axis=1)
        view_f1 = f1_score(y_test, view_preds, average="macro")
        logger.info(f"  {view_name:20s} Test Macro-F1: {view_f1:.4f}")

    # Ensemble test performance
    test_preds = ensemble.predict(test_views)
    test_f1 = f1_score(y_test, test_preds, average="macro")

    logger.info(f"\n{'='*40}")
    logger.info(f"Stacked Ensemble Test Macro-F1: {test_f1:.4f}")
    logger.info(f"{'='*40}")
    logger.info(f"\n{classification_report(y_test, test_preds, digits=4)}")
    logger.info(f"\n*** Final Test Macro-F1: {test_f1:.4f} ***")

    return test_f1


if __name__ == "__main__":
    main(task="T1")
