# Script Thuyết Trình - Beyond Binary: AI Code Detection

**Nhóm**: Đào Sỹ Duy Minh, Huỳnh Trung Kiệt, Trần Chí Nguyên và cộng sự (HCMUS_TheFangs)

---

## Slide 1: Title

Xin chào thầy, hôm nay nhóm em xin trình bày đề tài "Beyond Binary: Unified Detection and Attribution of AI-Generated Code via Hierarchical Family-Aware Learning." Đây là hướng nghiên cứu về phát hiện và truy nguồn code được tạo bởi AI, với mục tiêu NeurIPS 2026.

---

## Slide 2: Outline

Bài trình bày gồm 7 phần: từ vấn đề đặt ra, 3 benchmark sử dụng, method SpectralCode đã hoàn thành, HierTreeCode đang phát triển, kết quả trên 3 benchmark, và các insight từ 16+ thí nghiệm đã chạy.

---

## Slide 3: Existing Benchmarks are Binary-Only and Limited

Đầu tiên, hãy nhìn vào các benchmark hiện có. GPTSniffer chỉ có 7.4 nghìn mẫu, 1 ngôn ngữ, 1 model (ChatGPT). Whodunit còn nhỏ hơn --- 1.6 nghìn mẫu. CodeGPTSensor tuy lớn (1.1 triệu) nhưng chỉ cover 2 ngôn ngữ và 1 model. CodeMirage có 10 ngôn ngữ nhưng vẫn chỉ là binary classification.

TẤT CẢ các benchmark cổ điển này đều có chung hạn chế: BINARY ONLY, không có authorship attribution, không có OOD evaluation, không có adversarial data. Và quan trọng --- chúng đã BÃO HÒA, fine-tuned models đạt 98%+ F1.

Do đó nhóm chọn 3 benchmark mới: CoDET-M4 (ACL Findings 2025) có authorship 6 lớp, DroidCollection (EMNLP 2025) có adversarial DPO-tuned, và AICDBench (EACL 2026) có OOD 4-split progressive evaluation. Đây là 3 benchmark mà các benchmark cổ điển KHÔNG THỂ ĐO LƯỜNG ĐƯỢC.

---

## Slide 4: The Real Challenges Remain

3 vấn đề mở mà benchmark cổ điển không thể đánh giá:

1. AUTHORSHIP ATTRIBUTION: UniXcoder chỉ đạt 66.33% F1 trên 6 lớp --- giảm 32 điểm so với binary. Benchmark cổ điển không có nhãn đa lớp nên không thể evaluate bài toán này.

2. OOD GENERALIZATION: AICD T1 cho thấy val 99.5% nhưng test 29.8% --- SỤP ĐỔ thảm hại khi đổi ngôn ngữ. Benchmark cổ điển train và test trên cùng ngôn ngữ nên OOD là "vô hình."

3. ADVERSARIAL ROBUSTNESS: DROID có 157K mẫu adversarial DPO-tuned. DroidDetect recall giảm từ 98% xuống 92% trên adversarial code. Không benchmark cổ điển nào có dữ liệu adversarial.

---

## Slide 5: The Bigger Picture

Slide này trình bày sự tiến hóa của vấn đề theo thời gian:
- 2023: "Code này AI viết hay người viết?" --- ĐÃ GIẢI QUYẾT
- 2024: "Model nào viết?" --- Author attribution chỉ đạt 66% F1 với SOTA
- 2025: "Model có đánh lừa được detector không?" --- DPO-tuned adversarial code xuất hiện (DROID, EMNLP 2025)
- 2026: "Có generalize được sang ngôn ngữ mới không?" --- OOD collapse từ 99% xuống 30% (AICDBench, EACL 2026)

TẠI SAO attribution khó? Vì các LLM hình thành "cây gia phả." Nxcode là phiên bản fine-tune từ Qwen1.5, CodeLlama là fine-tune từ Llama. Các cặp cha-con này chia sẻ "DNA phong cách" gần như giống nhau --- dẫn đến 40% confusion giữa Qwen và Nxcode.

INSIGHT CỦA NHÓM: Cây gia phả của LLM không phải vấn đề --- mà là PRIOR KNOWLEDGE có cấu trúc. Models cùng family nên gần nhau nhưng vẫn phân biệt được trong embedding space.

---

## Slide 6: Three Benchmarks

Nhóm sử dụng 3 benchmark bổ sung cho nhau, tổng cộng 3.6 triệu mẫu code:

1. **CoDET-M4** (Orel et al., ACL Findings 2025): 500K mẫu, 3 ngôn ngữ, 5 generator. Mạnh nhất cho bài toán authorship attribution 6 lớp. Đây là nơi vấn đề Nxcode/Qwen confusion rõ nhất.

2. **DroidCollection** (Orel et al., EMNLP 2025): 1.06 triệu mẫu, 7 ngôn ngữ, 43 models. Đặc biệt có machine-refined code (code người được AI viết lại) và 157K mẫu adversarial DPO-tuned. Đây là benchmark duy nhất có dữ liệu adversarial thực sự.

3. **AICDBench** (Orel et al., EACL 2026): 2.05 triệu mẫu, 9 ngôn ngữ, 77 models. Benchmark lớn nhất với 4-split progressive OOD evaluation --- train trên 3 ngôn ngữ, test trên 9.

TẠI SAO 3 benchmark? Vì mỗi cái cover một khía cạnh khác nhau: CoDET-M4 cho attribution, DROID cho adversarial, AICD cho OOD. Gộp lại thì cover toàn bộ bài toán.

---

## Slide 7: SpectralCode Architecture Pipeline

Đây là toàn bộ pipeline của SpectralCode --- method chính đã hoàn thành của nhóm.

Architecture gồm 2 đường chính:
- **Neural Path** (xanh): gồm 3 encoder (ModernBERT, AST BiLSTM, Structural MLP) được fuse qua Cross-Attention
- **Spectral Path** (cam): FFT spectral analysis trên chuỗi token

2 đường này được kết hợp qua Learned Soft Gate để cho ra kết quả cuối cùng.

---

## Slide 8: Stream 1 --- ModernBERT Token Encoder

Stream đầu tiên là ModernBERT-base. Đây là pre-trained transformer encoder với 149 triệu tham số. Nó tokenize code nguồn và trả về [CLS] token representation ở R^768.

TẠI SAO ModernBERT? Vì nó outperform CodeBERT và RoBERTa trên code tasks, có SDPA attention nhanh gấp đôi, và đặc biệt --- paper DROID dùng ModernBERT-Large 395 triệu tham số, mà nhóm em chỉ dùng Base 149 triệu tham số nhưng vẫn đạt kết quả tương đương!

ModernBERT bắt được các pattern từ vựng và ngữ nghĩa --- ví dụ GPT có xu hướng dùng list comprehension, Llama thích explicit loops.

Tuy nhiên, token-level thôi thì chưa đủ --- nó bỏ lỡ các pattern cấu trúc (độ sâu lồng nhau, control flow) mà chỉ có thể thấy trong AST. Đó là lý do cần stream thứ 2.

---

## Slide 9: Stream 2 --- AST Encoder (tree-sitter + BiLSTM)

Stream thứ hai là AST encoder. AST --- Abstract Syntax Tree --- là biểu diễn cây của cấu trúc code. Mỗi node là một thành phần cú pháp: if_statement, function_definition, return_statement, v.v.

Ví dụ: dòng code "if x > 0: return x" sẽ được parse thành một cây với if_statement ở gốc, binary_expression và return_statement là con.

TREE-SITTER là gì? Là một parser nhanh, incremental, hỗ trợ 40+ ngôn ngữ lập trình. Nó phân tích code thành cây AST trong milliseconds. Nhóm em hỗ trợ 9 ngôn ngữ: Python, Java, C++, C, Go, PHP, C#, JavaScript, Rust.

QUY TRÌNH: Code được tree-sitter parse thành AST tree, sau đó duyệt DFS lấy chuỗi node types (67 loại trong vocabulary), rồi BiLSTM encode thành vector h_ast ở R^128. BiLSTM đọc chuỗi theo 2 chiều (trái sang phải và phải sang trái) để bắt được ngữ cảnh đầy đủ.

TẠI SAO AST cho authorship? Vì mỗi LLM có "phong cách cú pháp" riêng --- GPT dùng nhiều ternary expression lồng nhau, Llama thích if-else tường minh, Qwen hay dùng list comprehension. AST bắt được các "dấu vân tay" cấu trúc này mà token-level không thấy được.

---

## Slide 10: Stream 3 (Structural Features) + Stream 4 (Spectral FFT)

**22 Structural Features** là các metric phong cách viết code thủ công. Gồm 3 nhóm:
- Layout: độ dài dòng trung bình, indent mean/variance/max
- Complexity: số function, loop, conditional, try-catch
- Naming style: tỷ lệ snake_case vs camelCase, độ dài biến trung bình, tỷ lệ biến 1 ký tự

Đây là "fingerprint" phong cách lập trình mà mỗi LLM để lại.

**Spectral FFT Path** --- đây là ĐIỂM MỚI CHÍNH của SpectralCode.

Ý tưởng cốt lõi: Chuỗi token ID chính là một tín hiệu 1D rời rạc. Quá trình decoding của LLM (top-k, temperature sampling) tạo ra các MÔ HÌNH TẦN SỐ TUẦN HOÀN mà không nhìn thấy được ở không gian token.

FFT --- Fast Fourier Transform --- là công cụ toán học phân tách tín hiệu thành các thành phần tần số. Giống như khi phân tích âm thanh thành các note nhạc, FFT phân tích chuỗi token thành các tần số đặc trưng.

Nhóm áp dụng FFT ở 4 cửa sổ khác nhau (32, 64, 128, 256 token) để bắt pattern ở nhiều scale. Mỗi cửa sổ trích xuất 16 đặc trưng: 8 năng lượng dải tần (từ tần số thấp đến cao), spectral centroid (trọng tâm phổ), rolloff (tần số chứa 85% năng lượng), flatness (mức "phẳng" của phổ --- noise-like hay tonal), và peak frequency.

Tổng cộng 64 chiều, đưa qua MLP cho ra h_spec ở R^128.

Đây là kỹ thuật lấy cảm hứng từ spectral forensics trong phát hiện deepfake ảnh (Frank et al., ICML 2020), áp dụng lần đầu cho code detection.

---

## Slide 11: Fusion --- Cross-Attention + Learned Gate

3 representation của Neural Path (token R^768, AST R^128, structural R^128) được project về cùng chiều R^512, stack thành chuỗi 3 token, rồi đưa qua 4-head self-attention. Mỗi view "attend" tới các view khác --- ví dụ: token feature được điều chỉnh bởi AST structure, giúp model nhận ra "pattern token này đáng ngờ hơn với cấu trúc AST này."

Sau đó, Learned Soft Gate kết hợp Neural và Spectral predictions. Gate học trọng số cho từng mẫu --- không phải code nào cũng cần spectral như nhau. Ví dụ: code ngắn có thể chủ yếu dựa vào neural, code dài dựa nhiều hơn vào spectral.

Loss function gồm Focal Loss (giảm trọng số mẫu dễ, tập trung vào mẫu khó) trên 3 head: gate (chính), neural (phụ 0.3), spectral (phụ 0.3). Auxiliary losses đảm bảo cả 2 đường đều học representation hữu ích độc lập.

---

## Slide 12: CoDET-M4 Results

Kết quả trên CoDET-M4: SpectralCode đạt 99.06% F1 binary --- hơn UniXcoder (98.65%). Quan trọng hơn, trên bài toán author attribution 6 lớp, SpectralCode đạt 69.82% --- cao hơn UniXcoder 3.49 điểm, cao hơn CodeBERT 5.02 điểm.

Breakdown theo ngôn ngữ: C++ 72.31%, Java 70.12%, Python 69.22% --- khá ổn định. Nhưng breakdown theo nguồn code cho thấy GitHub chỉ 56.18% trong khi CodeForces đạt 77.17% --- chênh lệch 21 điểm. Đây là bottleneck chính --- code GitHub đa dạng phong cách hơn nhiều so với competitive programming.

---

## Slide 13: DroidCollection Results

Kết quả trên DROID (EMNLP 2025): SpectralCode đạt 88.77% weighted-F1 trên 3-class task --- THẮNG DroidDetect-Base (86.76%) và NGANG DroidDetect-Large (88.78%).

Đặc biệt, nhóm chỉ dùng ModernBERT-base (149M params) trong khi paper dùng Large (395M params), và chỉ train trên 1/10 dữ liệu (100K/1.06M). Đây là kết quả rất tích cực --- model nhỏ hơn, ít data hơn mà vẫn competitive. Khi scale lên full data, nhóm kỳ vọng sẽ vượt cả DroidDetect-Large.

---

## Slide 14: AICDBench Results

AICD là benchmark khó nhất --- 77 models, 9 ngôn ngữ, OOD evaluation nghiêm ngặt.

SpectralCode đạt 29.83% trên T1 (competitive với ModernBERT 30.61%), 56.31% trên T3. T2 (12-class family attribution) còn 18.93% --- đây là nơi HierTreeCode được kỳ vọng sẽ cải thiện với family-aware loss.

LƯU Ý QUAN TRỌNG: Tất cả methods --- kể cả SOTA --- đều bị OOD collapse nghiêm trọng trên AICD T1: val 99.5% nhưng test chỉ 29.8%. Đây là vấn đề cơ bản của benchmark --- train 3 ngôn ngữ, test 9 ngôn ngữ với domain hoàn toàn mới --- không phải của method.

Nhóm đang chỉ dùng 1/20 dữ liệu (100K/2.05M), nên còn nhiều room để cải thiện khi chạy full data.

---

## Slide 15: HierTreeCode Pipeline

HierTreeCode = SpectralCode + Hierarchical Affinity Loss. Pipeline giống hệt SpectralCode, chỉ thêm một nhánh MÀU ĐỎ: lấy h_neural và áp dụng Hierarchical Affinity Loss.

Family Tree được định nghĩa cho từng benchmark: Qwen -> {Qwen1.5, Nxcode}, Meta -> {Llama3.1, CodeLlama}, Google -> {Gemma, Gemini}.

Batch-hard triplet loss hoạt động như sau: với mỗi anchor sample, tìm mẫu cùng family xa nhất (hard positive) và mẫu khác family gần nhất (hard negative). Loss enforce: khoảng cách positive + margin < khoảng cách negative. Margin alpha = 0.3, dùng cosine distance trên L2-normalized embeddings.

Khác biệt duy nhất so với SpectralCode: thêm 0.4 * L_hier vào total loss. Mọi thứ khác giữ nguyên.

---

## Slide 16: HierTreeCode Preliminary Results

Trên CoDET-M4, HierTreeCode đạt 70.55% author F1 --- cao hơn SpectralCode 0.73 điểm và cao hơn UniXcoder 4.22 điểm. Gain tập trung vào Qwen1.5 (+3% F1) --- đúng là class khó nhất.

Nhóm kỳ vọng HierTreeCode sẽ cải thiện hơn nữa trên:
- DROID T3/T4: family hierarchy (human → refined → machine → adversarial) giúp phân biệt các loại code
- AICD T2: 12-class family attribution --- perfect match cho family-aware loss, đặc biệt nhóm Google (Gemma+Gemini) và Chinese AI (DeepSeek+Qwen)

Code đã sẵn sàng, đang chạy experiments.

---

## Slide 17: Evaluation Plan + Timeline

Tổng kết trạng thái hiện tại:
- CoDET-M4: ĐÃ XONG binary 99.06%, author 70.55%
- DROID: ĐÃ XONG T3 88.77%, T4 88.02% (SpectralCode)
- AICD: ĐANG CHẠY --- kết quả sơ bộ competitive

Timeline: Tháng 4-6 chạy OOD CoDET + DROID HierTreeCode, tháng 6-8 AICD full eval + full data, tháng 8-10 viết paper và submit NeurIPS 2026.

---

## Slide 18: Contributions

4 đóng góp chính:
1. SpectralCode: multi-stream architecture (token + AST + structural + spectral FFT) với cross-attention fusion và learned gating --- đã SOTA trên CoDET-M4 và thắng DroidDetect-Base trên DROID
2. HierTreeCode: lần đầu sử dụng cây gia phả LLM làm prior cho code attribution --- encode quan hệ cha-con thành ràng buộc khoảng cách trong embedding space
3. Unified evaluation trên 3 benchmark (3.6M mẫu) --- đầu tiên cover binary + attribution + OOD + adversarial trong cùng một method
4. Insight về scaling: kết quả competitive với chỉ 1/10 - 1/20 data, model base nhỏ hơn 2.6x so với paper mà vẫn ngang điểm

---

## Slide 19: CoDET-M4 Experiment Insights (10 methods)

Nhóm đã explore 10 methods khác nhau trên CoDET-M4. Leaderboard cho thấy các method cluster trong khoảng 69.7-70.6% author F1. HierTreeCode dẫn đầu với 70.55%.

NEGATIVE RESULT QUAN TRỌNG: DANN (domain adversarial) THẤT BẠI thảm hại --- từ 69.82% xuống 62.89%. Gradient reversal ép model tạo feature không phân biệt generator --- nhưng đây CHÍNH LÀ điều ta KHÔNG muốn trong attribution. Qwen1.5 F1 rơi xuống 0.198 (random). Bài học: domain adversarial KHÔNG phù hợp cho attribution --- đây là negative result có giá trị cho cộng đồng.

Ngoài ra: GAT trên AST cũng không giúp (GraphStyleCode thậm chí kém hơn baseline), trong khi các method đơn giản hơn như RAGDetect (retrieval-augmented) và BiScopeCode (MLM memorization probe) lại hiệu quả.

---

## Slide 20: AICD + DROID Insights (13 methods)

Trên AICD + DROID, nhóm chạy 13 method. SpectralCode dẫn đầu overall (0.549 avg macro-F1), TokenStat mạnh nhất trên DROID (0.852).

5 insight chính:
1. AICD T1 OOD vẫn chưa ai giải được --- tất cả methods collapse từ 99% xuống 30%
2. Spectral features transfer tốt nhất dù kiến trúc đơn giản --- FFT bắt được các pattern tần số mà neural features bỏ lỡ
3. Token statistics (entropy, burstiness, Yule-K) cực mạnh trên DROID --- cho thấy statistical features vẫn có giá trị
4. Embedding Mixup giúp OOD --- DomainMix đạt best AICD T1 nhờ data augmentation
5. Chỉ dùng 1/10-1/20 data --- full data sẽ cho kết quả cao hơn, đây là low-hanging fruit

Các method thất bại: CausAST (orthogonal penalty triệt tiêu hoàn toàn), OSCP (whitening loss quá lớn ~206, lấn át loss chính), AST-IRM (IRM penalty nổ tới 5000).

---

## Slide 21: Thank You

Cảm ơn thầy đã lắng nghe. Nhóm sẵn sàng trả lời câu hỏi. 

Ba benchmark sử dụng: CoDET-M4 (ACL Findings 2025), DroidCollection (EMNLP 2025), AICDBench (EACL 2026). Code và experiments có sẵn.

---

**GHI CHÚ CHO THUYẾT TRÌNH:**
- Thời gian dự kiến: 20-25 phút
- Tập trung vào story: benchmark cũ binary-only đã bão hòa → cần attribution/OOD/adversarial → 3 benchmark mới → SpectralCode đã có kết quả tốt → HierTreeCode là bước tiếp theo
- Nhấn mạnh: model base (149M) thắng model large (395M), chỉ dùng 1/10 data
- Nhấn mạnh negative results: DANN failure là insight có giá trị cho cộng đồng nghiên cứu
- Khi giải thích FFT: dùng analogy âm nhạc --- "giống như phân tích bài hát thành các note, FFT phân tích chuỗi token thành các tần số đặc trưng cho từng LLM"
- Khi giải thích AST: vẽ trên bảng nếu cần --- "if x > 0: return x" → cây với if ở gốc
