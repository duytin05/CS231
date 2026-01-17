# CS231 - Nhập môn Thị giác máy tính (Computer Vision)

## **Đồ án: Hệ thống Nhận diện một số loại Nông sản Tự động**

---

##  Giới thiệu

Việc thực hiện đề tài này xuất phát từ hai nhu cầu cấp thiết: **thực tiễn ngành bán lẻ** và **nghiên cứu học thuật** về các kiến trúc mạng nơ-ron hiện đại.

### 1. Về mặt thực tiễn: Chống gian lận thương mại
Hệ thống nhằm giải quyết bài toán chống thất thoát doanh thu tại các khu vực tự thanh toán (self-checkout) của siêu thị. 
* **Vấn đề:** Hình thức gian lận "Tráo đổi nhãn" (khách hàng chọn mã sản phẩm giá rẻ cho mặt hàng giá cao) đang gây thiệt hại lớn cho các hệ thống bán lẻ.
* **Giải pháp:** Xây dựng mô hình AI nhận diện khách quan và chính xác loại sản phẩm thực tế. Đây là thành phần cốt lõi của hệ thống đối soát tự động, giúp phát hiện sai lệch thông tin và ngăn chặn gian lận hiệu quả trong giờ cao điểm.

### 2. Về mặt công nghệ và học thuật: CNNs vs. ViTs
Đồ án tập trung nghiên cứu và so sánh kiểm chứng hiệu năng giữa hai trường phái kiến trúc mạng nơ-ron tiên tiến nhất hiện nay:
* **CNNs (ConvNeXt V2):** Thế hệ mới của mạng tích chập truyền thống với các cải tiến về hiệu suất xử lý.
* **Vision Transformers (Swin Transformer V2):** Mô hình sử dụng cơ chế Attention hiện đại đang là tâm điểm của cộng đồng nghiên cứu.
* **Mục tiêu:** Đánh giá xem liệu kiến trúc Transformer mới mẻ có thực sự vượt trội hơn CNN truyền thống trong việc phân loại 50 loại trái cây, đặc biệt là khả năng chịu lỗi trong môi trường giả lập nhiễu thực tế.

---


##  Thành viên nhóm
| STT | MSSV | Họ và tên | Github |
|---|---|---|---|
| 1 | 23521592 | Đỗ Lê Duy Tín | [duytin05](https://github.com/duytin05) |

---

##  Dữ liệu (Dataset)
* **Bộ dữ liệu:** Trích xuất từ bộ dữ liệu **Fruits-360**.
* **Số lượng:** Phân loại **50 loại trái cây**, tổng cộng khoảng **32,146 hình ảnh** thực nghiệm.


##  Phương pháp (Methodology)

Dự án thực hiện nghiên cứu và thí nghiệm trên các kiến trúc Deep Learning hiện đại nhằm tối ưu hóa khả năng nhận diện trái cây trong điều kiện thực tế:

* **Mô hình kiến trúc (Architectures):**
    * **Swin Transformer V2:** Sử dụng cơ chế Window Attention giúp nắm bắt đặc trưng phân cấp của hình ảnh hiệu quả hơn so với CNN truyền thống.
    * **ConvNeXt V2:** Cải tiến từ kiến trúc CNN thuần túy với các kỹ thuật từ Transformer như FCMAE và GRN để tăng hiệu suất xử lý ảnh.
* **Kỹ thuật tiền xử lý (Preprocessing):**  Resize ảnh về kích thước $192 \times 192$ (SwinV2) và $224 \times 224$ (ConvNeXt) để phù hợp với input của pre-trained models.
    

* **Giao diện Demo:** Sử dụng thư viện **Gradio** để xây dựng ứng dụng web cho phép người dùng tải ảnh và nhận kết quả phân loại thời gian thực.



---

##  Kết quả (Results)

Sau quá trình huấn luyện và đánh giá trên bộ dữ liệu 50 lớp trái cây, các mô hình đạt được kết quả ấn tượng:

| Mô hình | Accuracy | Macro F1-Score | Đặc điểm nổi bật |
|---|---|---|---|
| **Swin Transformer V2** | **0.9731** | **0.9750** | Độ chính xác cao nhất, nhận diện tốt các lớp tương đồng. |
| **ConvNeXt V2** | 0.9496 | 0.9482 | Tốc độ suy luận nhanh, hiệu quả trên thiết bị cấu hình trung bình. |

### Nhận xét:
* Mô hình Swin Transformer V2 thể hiện sự vượt trội trong việc phân biệt các loại quả có độ tương đồng cao như Táo (Braeburn vs Pink Lady) nhờ vào cơ chế Attention tập trung vào các chi tiết bất biến.
* Kết quả ma trận nhầm lẫn (Confusion Matrix) cho thấy tỉ lệ phân loại sai giữa các lớp là cực kỳ thấp.



---

##  Cài đặt & Hướng dẫn sử dụng

### Bước 1: Clone dự án
```bash
git clone [https://github.com/duytin05/CS231.git](https://github.com/duytin05/CS231.git)
cd CS231
```
### Bước 2: Tải trọng số mô hình (Model Weights)
Do kích thước file lớn (>600MB), vui lòng tải thủ công từ Google Drive và đặt vào thư mục gốc của dự án:

* [👉 **Tải SwinV2 & ConvNeXtV2 Weights**](https://drive.google.com/drive/folders/1NZYMRymOolTq6XM0BrhZuebRkmn-J1zg?usp=sharing)

### Bước 3: Cài đặt thư viện cần thiết
Mở Terminal/CMD tại thư mục dự án và chạy lệnh sau:
```bash
pip install -r requirements.txt
```
### Bước 4: Khởi chạy ứng dụng Demo
Chạy file giao diện bằng lệnh:

```bash
python app.py
```
---

##  Tham khảo

* **Bộ dữ liệu:** [Fruits-360 Dataset on Kaggle](https://www.kaggle.com/datasets/moltean/fruits) - Tác giả: Mihai Oltean.
* **Swin Transformer V2:** [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883) - Ze Liu et al.
* **ConvNeXt V2:** [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808) - Sanghyun Woo et al.
* **Giao diện Demo:** [Gradio Documentation](https://gradio.app/docs/) - Thư viện hỗ trợ xây dựng giao diện cho mô hình máy học.
