# Kiểm tra bộ nộp Lab 22

Ngày kiểm tra: 2026-10-09.

## Đã xác nhận

- Họ tên Nguyễn Vũ Quang Anh và khoá/lớp 4-L3B-Track3-2A202602805 đã được điền trong REFLECTION.md.
- Notebook `colab/Lab22_DPO_T4.ipynb` có 141 cell (thêm một cell markdown đọc mẫu local), 57 cell có execution count và 54 cell có output; không có output traceback lỗi.
- NB0 đã in `✓ Khớp tham chiếu: 0.6981`, kiểm tra khởi tạo in loss 0,6931 và reward bằng 0.
- SFT: 1.000 mẫu, 1 epoch, 125 bước, loss trung bình 1,3603; có output lưu adapter và mô hình merged.
- NB2: 800 train / 100 eval, `D.assert_disjoint` chạy không lỗi; chosen dài hơn ở 65,875% cặp. Hash các file Parquet vẫn khớp split.json. Đọc lại toàn bộ prompt trong hai file tại máy local xác nhận 0 câu trùng nhau sau khi chuẩn hoá.
- Đã trích xuất đủ ba cặp thật (hàng 0–2) từ train.parquet; nội dung và nhận xét có ở markdown NB2, PREFERENCE_REVIEW.md và preference_samples.json. Hai mẫu bổ sung được lấy local sau khi Colab bị ngắt, không tạo output thực thi giả và không thay đổi dữ liệu.
- NB3: 100 bước, vòng train 27 phút 20 giây; margin held-out +0,085639, reward accuracy 69%, diagnosis INTENDED.
- NB4: 8 câu cố định + 50 held-out; hai RM sanity 100%. JSON trong output notebook khớp hoàn toàn judge_summary.json; hash của side_by_side.jsonl khớp kết quả chấm. Win rate held-out 49%, CI [42%; 57%].
- Có đủ bốn ảnh bắt buộc, các file cấu hình và số liệu DPO, dữ liệu preference, đầu ra NB4 và REFLECTION.md.
- NB3b: có output cả năm biến thể, 300 train / 100 eval / 20 probe, summary và biểu đồ. Phản tư có bảng và phân tích bonus này.
- Đã cập nhật REFLECTION.md từ notebook, bỏ các ghi chú cũ về thiếu notebook/thiếu thông tin người làm; vẫn giữ thông tin chưa đo thay vì tự điền số.

## Điểm còn cần hoàn thiện

1. **Chưa chạy được make verify/git status tại máy local** vì terminal lỗi khởi động `helper_unknown_error: setup refresh had errors`. Chưa xác nhận commit, push hoặc repo public.
2. Chi phí và peak VRAM chưa có thông tin; phản tư ghi rõ chưa xác nhận. Những mục này không được tự suy ra từ thông báo OOM.

## Lưu ý khi chạy verify trên máy khác

Verifier hiện so sánh đường dẫn reference tuyệt đối với repo hiện tại. Adapter giữ đường dẫn thật `/content/lab22/models/sft-merged`, nên verifier trên Windows có thể báo WRONG REF sau khi chuyển repo. Không sửa metadata huấn luyện để che đường dẫn; cần kiểm tra tính di chuyển của verifier hoặc chạy trong môi trường Colab phù hợp.

NB5–NB7, beta-sweep, chấm chéo API và HF Hub chưa có bằng chứng hoàn thành và không được đánh dấu nhận bonus. Trọng số vẫn được loại khỏi Git; stats.json và config.json nhỏ cần cho bằng chứng đã được cho phép commit trong .gitignore.
