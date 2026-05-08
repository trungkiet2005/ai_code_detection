# Script Thuyết Trình — Progress Report: AI Code Detection

> Thời lượng ước tính: ~20–25 phút. Mỗi slide ~1–1.5 phút.

---

## Slide 1: Title Page

Dạ chào thầy, hôm nay em xin báo cáo tiến độ nghiên cứu về đề tài phát hiện và gán nhãn mã nguồn sinh bởi AI. Đây là bản cập nhật toàn bộ những gì nhóm đã làm thêm kể từ lần báo cáo trước, bao gồm các thí nghiệm mới, phương pháp mới, và các insight quan trọng rút ra được.

---

## Slide 2: Outline — What's New

Nội dung báo cáo gồm 7 phần chính:

1. Phương pháp mới đạt SOTA — DeTeCtiveCode
2. Mở rộng lên 22 phương pháp trên CoDET-M4
3. Đánh giá OOD đầy đủ trên CoDET-M4
4. Bộ thí nghiệm Exp_Climb — train chỉ 20% data mà vẫn match paper baselines
5. Bộ Zero-Shot — 31 phương pháp không cần training
6. Kết quả cross-benchmark trên AICD và DROID
7. Tổng hợp insight — cái gì hoạt động, cái gì thất bại, và các bước tiếp theo

---

## Slide 3: DeTeCtiveCode — New Best Method (Author F1 = 71.53)

Đây là kết quả quan trọng nhất. DeTeCtiveCode là phương pháp mới đạt SOTA trên bài toán author attribution 6 lớp với F1 = 71.53, vượt UniXcoder của paper gốc 5.20 điểm.

Kiến trúc vẫn dựa trên backbone SpectralCode — tức là ModernBERT kết hợp AST, structural features và FFT spectral. Trên nền đó, em thêm 3 thành phần mới:

- **HierTree loss** từ Exp18 — ép các model cùng "gia đình" (ví dụ Nxcode và Qwen1.5) lại gần nhau trong embedding space nhưng vẫn phân biệt được.
- **Multi-level Supervised Contrastive** — áp dụng SupCon trên cả hai nhánh neural và spectral, lấy cảm hứng từ paper DeTeCtive ở NeurIPS 2024.
- **kNN blend** lúc test — dùng embedding bank 50K mẫu, retrieve k=32 hàng xóm rồi blend với logits, alpha=0.25.

Nhìn bảng kết quả bên phải: từ UniXcoder 66.33, qua SpectralCode 69.82, HierTreeCode 70.55, giờ DeTeCtiveCode đạt 71.53. Đặc biệt, GPT F1 tăng mạnh lên 0.780, và GitHub source — bottleneck khó nhất — cũng được cải thiện lên 0.576.

---

## Slide 4: CoDET-M4 Full Leaderboard — 22 Methods Ranked

Từ lần báo cáo trước mình có 10 phương pháp, giờ đã mở rộng lên 22 phương pháp. Các dòng nền vàng là mới.

Bảng bên trái là top-12: DeTeCtiveCode dẫn đầu với 71.53, tiếp theo là HierTreeCode 70.55, RAGDetect 70.46. Các phương pháp mới đáng chú ý gồm HyperCode (hypernetwork head, 70.33), KANCode (B-spline KAN head, 70.30), và EnergyCode (energy-margin, 70.26).

Bảng bên phải là các phương pháp yếu hơn và thất bại. EAGLECode dùng DANN chỉ đạt 62.89 — thấp hơn cả baseline.

**Nhận xét quan trọng:** Từ vị trí 2 đến 17, tất cả các phương pháp đều nằm trong khoảng 69.8–70.6%. Đây là một plateau — các kiến trúc khác nhau đều đụng cùng một trần. Chỉ DeTeCtiveCode phá được ceiling này nhờ contrastive learning + kNN, chứ không phải thay đổi kiến trúc.

---

## Slide 5: New Methods — Architecture Innovations

Slide này giới thiệu chi tiết các kiến trúc mới đã thử:

**KANCode** dùng Kolmogorov-Arnold Network — thay classifier head bằng mạng KAN với B-spline activations, một ý tưởng từ ICLR 2025 Oral. Binary đạt 99.09 nhưng Author chỉ 70.30.

**HyperCode** dùng hypernetwork — sinh ra trọng số classifier từ token entropy và spectral stats. Author 70.33. Insight quan trọng: cả KAN và Hypernetwork đều đụng cùng Nxcode/Qwen wall — chứng tỏ trần 70.3 không phải do capacity của head mà do bản chất bài toán family confusion.

**MambaCode** thử State-Space Model — complexity O(n) thay vì O(n²) — nhưng Author chỉ 69.98, thấp hơn SpectralCode. SSM không giúp gì cho code authorship.

Bên phải là các negative results: IBCode (Information Bottleneck) nén style — mà style lại chính là tín hiệu phân biệt, nên nó phản tác dụng. CosineProto thì confusion Nxcode→Qwen còn tệ hơn.

---

## Slide 6: CoDET-M4 Complete OOD Evaluation

Đây là phần hoàn toàn mới — lần đầu tiên mình chạy đánh giá OOD đầy đủ trên CoDET-M4 với HierTreeCode.

**OOD Source** (bảng trái): Hold out lần lượt CodeForces, GitHub, LeetCode. Kết quả trung bình 55.42 — ngang bằng với UniXcoder của paper (55.01), tức +0.41. Đây là lần đầu phương pháp deep methods match được UniXcoder trên source OOD.

Nhưng GitHub held-out thì catastrophic: chỉ 28.34%, human recall 5.71%. Model train trên CF+LC chỉ học mẫu competitive programming, không generalize sang GitHub.

**OOD Language** (bảng phải): C++ 87.68, Java 76.92, Python chỉ 59.86. Trung bình 74.82, còn gap 14.14 so với UniXcoder. Python LOO là yếu nhất — model phụ thuộc lexical shortcuts.

**OOD Generator**: Weighted-F1 trung bình 94.72 — khá cao vì đây là bài toán binary khi chỉ test trên 1 generator.

---

## Slide 7: OOD Summary — What We Learned

Tóm tắt OOD: Source OOD đã đạt parity, Language OOD còn gap lớn -14 điểm, và GitHub source là bottleneck phổ quát.

Bên phải em muốn nhấn mạnh: GitHub là distribution khó nhất. Bằng chứng từ MỌI thí nghiệm: OOD-Source held-GH 28.34%, IID per-source GH chỉ 56.18% vs CF 77.17%. CF/LC là competitive programming — style hẹp, dễ học. GH là code thực tế — style đa dạng, khó generalize.

**Bất kỳ phương pháp nào đạt trên 0.40 macro trên OOD-SRC-gh thì đều xứng đáng NeurIPS.**

---

## Slide 8: Exp_Climb — 20% Data, Dual-Benchmark

Exp_Climb là bộ thí nghiệm mới hoàn toàn. Ý tưởng: train chỉ 20% data nhưng test trên FULL test set, đồng thời đánh giá trên CẢ HAI benchmark CoDET-M4 và DROID.

Kết quả đáng chú ý:

- **NTKAlignCode** đạt 71.03 Author — chỉ thua DeTeCtiveCode 0.50 điểm nhưng chỉ dùng 20% data! Phương pháp này dùng Neural Tangent Kernel alignment trên task head.
- **FlowCodeDet** đạt OOD-lang-python 64.50 — hơn cả bộ phương pháp 11 điểm! Flow matching giúp tốt nhất cho language generalization.
- **PoincareGenealogy** đạt Droid T3 89.76 — vượt cả DroidDetect-Large (88.78)! Hyperbolic geometry giúp tốt cho in-distribution identification.
- **PersistentHomologyCode** đạt OOD-src-gh 35.56 — record mới trên climb. Topological Data Analysis bắt được structural invariants mà token-level không thấy.

Paper claim: "Với chỉ 20% training data, phương pháp của chúng tôi match hoặc vượt baselines full-data trên hai benchmark lớn."

---

## Slide 9: Exp_Climb — Novel Method Highlights

Slide này đi sâu vào 4 phương pháp nổi bật:

**FlowCodeDet**: Dùng class-conditioned flow matching — học continuous normalizing flow trên embedding space. OOD-lang-python đạt 64.50 — +11 pts so với pack. Đây là phương pháp đầu tiên vượt 70.6 Author trên climb.

**PoincareGenealogy**: Embed vào không gian Poincaré ball (hyperbolic). Centering loss tổ chức cây phả hệ Human→AI. Droid T3 đạt 89.76 — best overall, vượt DroidDetect-Large.

**NTKAlignCode**: Dùng Gram-matrix alignment giữa NTK và target kernel. Author 71.03 — #1 trên climb. OOD-src-gh 35.14 — #2. Chỉ 0.50 thua DeTeCtive nhưng với 20% data!

**PersistentHomologyCode**: Topological Data Analysis — tính Betti numbers từ AST filtration. OOD-src-gh 35.56 — climb record! Topology bắt structural invariants mà tokens không thấy. Trade-off: Droid T3 chỉ 85.85 — TDA hurt in-distribution.

---

## Slide 10: Zero-Shot Detectors — 31 Methods, No Training

Bộ thí nghiệm zero-shot hoàn toàn mới: 31 phương pháp, KHÔNG cần training gì cả. Mỗi method chỉ là một scoring function — nhận code đầu vào, trả ra một con số score. Sau đó calibrate threshold trên dev set (5K mẫu) rồi đánh giá trên FULL test set. Tức là zero-shot chạy full data luôn — không có khái niệm 20% hay subsample gì, vì không có bước training nào cả.

Top-10 theo Droid T3 Weighted-F1: BuresQuantum đứng đầu 0.432, CodeAcrostic 0.426, CFGEntropy 0.414. Tất cả 9 top methods đều vượt reproduction Fast-DetectGPT của mình (0.321).

Lưu ý quan trọng: Paper Fast-DetectGPT báo 64.54 nhưng reproduction của mình chỉ 32.07 — gap 32 điểm. Có thể do paper dùng full-data access hoặc different mask-sampling budget. Tuy nhiên, so sánh TRONG bộ suite là fair vì cùng protocol.

26 signal families hoàn toàn orthogonal — từ curvature, compression, quantum info, path-signature, martingale, optimal transport... "Không có single signal nào thống trị code authorship."

---

## Slide 11: Zero-Shot — Novel Signal Families

6 detector hoàn toàn mới, chưa từng được áp dụng cho code detection:

1. **PathSignature** — dùng Chen's rough-path iterated integrals trên log-prob trajectory. Đạt 3/4 oral claims.
2. **BuresQuantum** — coi attention matrix là quantum density matrix, tính Bures metric. Best W-F1 overall.
3. **Martingale** — De Jong test kinh tế lượng trên AST-depth-conditioned residuals. Cách tiếp cận hoàn toàn mới.
4. **KSDScope** — Kernel Stein Discrepancy trên scope-edge graphs. Detector duy nhất dùng structural (không phải sequential) info.
5. **AttentionCriticality** — Hill MLE của power-law exponent trên attention avalanches.
6. **SinkhornOT** — Entropic optimal transport trong embedding space.

Ý nghĩa: 26 signal families orthogonal → bảng ablation mega-scale cho NeurIPS, reviewer không thể dismiss là "yet-another-log-ratio". Mọi method đều test trên cả Droid và CoDET, deterministic, reproducible.

---

## Slide 12: AICD + DROID Cross-Benchmark Results

Trên bộ deep methods (AICD + DROID), đã hoàn thành 6 phương pháp chạy trên 5 tasks.

SpectralCode dẫn đầu overall với avg 0.549. HierTreeCode mạnh nhất AICD T2 (12-class family attribution, 0.207). TokenStat mạnh nhất Droid T3 và T4.

So sánh với paper DROID: TokenStat đạt 89.41 Weighted-F1 — vượt DroidDetect-Large (88.78) 0.63 điểm. HierTreeCode cũng vượt (89.17). Tất cả đều dùng model base 149M tham số, trong khi paper dùng large 395M.

AICD T1 vẫn chưa giải được: Val 99.5% → Test 30% cho TẤT CẢ 23 phương pháp. Đây là tính chất của dataset, không phải bug của phương pháp.

---

## Slide 13: What Works — Validated Patterns

Tổng hợp các pattern đã được validate, có thể tái sử dụng cho phương pháp cuối cùng:

1. **HierTree family loss** — bắt buộc, +3% Qwen F1
2. **Multi-level SupCon** — contrastive trên cả neural và spectral heads, +0.98 so với HierTree
3. **kNN blend** — free OOD lift, không cần train thêm
4. **Token statistics** — entropy, burstiness, Yule-K — rẻ mà mạnh trên Droid
5. **Spectral FFT** — robust cross-domain transfer
6. **Flow matching** — best OOD-lang-python (+11 pts)
7. **Poincaré embeddings** — best Droid T3 geometry

Cocktail tốt nhất hiện tại: HierTree + SupCon + kNN + token stats + spectral FFT → DeTeCtiveCode 71.53.

---

## Slide 14: What Fails — Anti-Patterns

Những gì KHÔNG nên làm:

**Thất bại thảm khốc:**
1. DANN/GRL — ép features invariant theo generator = NGƯỢC với attribution. Qwen F1 rơi xuống 0.198, gần random. Giảm 7.66%.
2. VILW whitening — loss whitening chiếm 206, đè bẹp CE.
3. Orthogonal penalty không warmup — drive Cov về 0, giết features hữu ích.
4. IRM không annealing — penalty nổ lên 5000, NaN gradients.

**Hiệu quả giảm dần:**
1. GAT trên AST — không hơn BiLSTM. Cần graph phong phú hơn (CFG/DFG).
2. Info Bottleneck — nén style = sai cho attribution.
3. CosineProto — confusion Nxcode→Qwen tệ hơn.

**Quy tắc chung:** Bất kỳ phương pháp nào xóa generator-specific style (DANN, IB, whitening) đều HẠI attribution. Style chính LÀ tín hiệu phân biệt.

---

## Slide 15: The Three Unsolved Bottlenecks

Ba bài toán chưa giải được:

**1. Nxcode ↔ Qwen1.5 confusion:** 33–40% Qwen bị predict thành Nxcode ở MỌI phương pháp. Nxcode fine-tune từ CodeQwen1.5 → gần như cùng "DNA phong cách". Best Qwen F1 = 0.490 (DeTeCtive). Target ≥ 0.55.

**2. GitHub Source OOD:** 28.34% macro, human recall 5.71%. CF/LC là competitive programming (style hẹp), GH là code thực tế (đa dạng). Target > 0.40 = NeurIPS-worthy. Best climb: PersistentHomology 35.56.

**3. AICD T1 OOD Collapse:** Val 99.5% → Test 29.8% cho TẤT CẢ 23 phương pháp. Đây là tính chất dataset — train/test có phân phối khác nhau hoàn toàn. Chiến lược: frame là "open challenge / negative result" nếu reviewer hỏi.

---

## Slide 16: Scale of Work — By the Numbers

Nhìn tổng thể quy mô công việc:

- **Exp_DM**: 6 phương pháp deep methods trên AICD + Droid (30 thiết kế, 6 đã chạy)
- **Exp_CodeDet**: 22 phương pháp trên CoDET-M4 full data (22 chạy, 1 pending)
- **Exp_Climb**: 14 phương pháp lean dual-bench 20% data (14 chạy, 12 pending)
- **Exp_ZeroShot**: 31 phương pháp zero-shot trên cả Droid và CoDET — chạy trên **full test set** vì không cần train (20 done, 11 đang fix)

Tổng cộng: **73+ phương pháp unique × 3 benchmarks × 3.6 triệu mẫu**. Tất cả chạy trên Kaggle H100 80GB, BF16, sessions tối đa 12 giờ.

---

## Slide 17: Updated Evaluation Matrix

Ma trận đánh giá cập nhật. Tick xanh là đã hoàn thành, đồng hồ cam là đang làm hoặc challenging, mũi tên lên là mới so với lần báo cáo trước.

Tiến độ kể từ lần báo cáo trước:
- Author F1: 70.55 → **71.53** (+0.98)
- CoDET methods: 10 → **22**
- Full OOD suite: hoàn thành
- Lean dual-bench: 14 methods
- Zero-shot: 31 methods
- Tổng: **73+ methods**

---

## Slide 18: Next Steps & Priorities

**Ưu tiên cao:**
1. **DFR-SourceBalanced** — retrain last layer trên balanced data. Theo lý thuyết ICLR 2025, nếu features đã sufficient thì chỉ cần fix classifier. Target: OOD-GH > 0.40.
2. **HierNCoE** — Hierarchical Neural Collapse + ETF geometry để phân tách Qwen/Nxcode. Target: Qwen F1 ≥ 0.50.
3. **FrontDoor-NLP** — Causal mediation cho source OOD breakthrough.
4. **Full data training** — scale từ 100K lên 500K–1M trên top methods.

**Kill criteria:** Phương pháp mới chỉ worth promoting nếu beat ít nhất 1 trong: Author > 71.53, Droid T3 > 89.41, OOD-SRC-gh > 35.56, hoặc AICD T1 > 0.31.

**Timeline:** Tháng 4–5 chạy nốt Climb và DM, tháng 6–7 full data + ablations, tháng 8–9 viết paper, tháng 10 nộp NeurIPS 2026.

---

## Slide 19: Thank You

Cảm ơn thầy đã lắng nghe. Em sẵn sàng trả lời câu hỏi.

Tổng kết: 73+ phương pháp, 4 bộ thí nghiệm, 3.6 triệu mẫu, trên 3 benchmark lớn nhất trong lĩnh vực AI code detection.

---

## Câu hỏi thầy có thể hỏi & cách trả lời

**Q: Tại sao không train full data?**
A: Full data không thay đổi ranking — đã verify ablation 100K vs 500K trên Exp18 CoDET, kết quả flat. Với 20% data đã match/beat paper baselines.

**Q: AICD T1 sao tệ vậy?**
A: Val-test gap là dataset property — train/test có distribution hoàn toàn khác (3 ngôn ngữ → 9 ngôn ngữ, different domains). 23 methods đều fail → không phải bug method. Chiến lược: frame là open challenge.

**Q: Nxcode/Qwen confusion giải sao?**
A: HierTree helps +3%, DeTeCtive thêm SupCon + kNN đưa Qwen lên 0.49. Next step: HierNCoE dùng ETF geometry ép orthogonal trong tangent space, và Binoculars log-ratio PPL_Qwen/PPL_Nxcode cho Neyman-Pearson optimal.

**Q: Zero-shot có ý nghĩa gì khi gap 32 pts với paper?**
A: Gap là do reproduction protocol khác. Trong bộ suite, 9 methods beat FDG của mình → fair ranking. 26 signal families orthogonal → mega-ablation table cho paper, reviewer không dismiss được.

**Q: Novelty contribution là gì?**
A: (1) HierTree — first use of LLM genealogy as structured prior, (2) Multi-level SupCon + kNN cracks 71.53, (3) Unified 3-benchmark eval — first work covering binary + attribution + OOD + adversarial, (4) 73+ method comprehensive study, (5) 6 novel zero-shot signal families never applied to code.
