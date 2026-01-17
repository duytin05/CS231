import gradio as gr
import torch
import torch.nn.functional as F
import os
from torchvision import transforms
from PIL import Image
from transformers import AutoImageProcessor, Swinv2ForImageClassification

# --- CẤU HÌNH ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/swinv2-base-patch4-window12-192-22k"
# Đường dẫn tương đối (để file .pth cùng thư mục với file .py này)
MODEL_PATH = "best_swin_v2.pth" 

# --- 1. DANH SÁCH 50 LỚP (HARDCODED) ---
class_names = sorted([
    'Apple Braeburn 1', 'Apple Crimson Snow 1', 'Apple Golden 1', 'Apple Granny Smith 1', 
    'Apple Pink Lady 1', 'Apple Red 1', 'Apple Red Delicious 1', 'Apple Red Yellow 1',
    'Apricot 1', 'Avocado 1', 'Avocado ripe 1', 'Banana 1', 'Clementine 1', 
    'Cucumber 1', 'Ginger Root 1', 'Grapefruit Pink 1', 'Grapefruit White 1', 
    'Lemon 1', 'Limes 1', 'Mandarine 1', 'Mango 1', 'Mango Red 1', 'Mangostan 1', 
    'Nectarine 1', 'Orange 1', 'Papaya 1', 'Passion Fruit 1', 'Peach 1', 'Peach Flat 1', 
    'Pear 1', 'Pear Abate 1', 'Pear Forelle 1', 'Pear Kaiser 1', 'Pear Monster 1', 
    'Pear Williams 1', 'Pepper Green 1', 'Pepper Red 1', 'Pepper Yellow 1', 'Pineapple 1', 
    'Pitahaya Red 1', 'Plum 1', 'Pomegranate 1', 'Potato Red 1', 'Potato White 1', 
    'Tangelo 1', 'Tomato 1', 'Tomato Cherry Red 1', 'Tomato Heart 1', 'Tomato Maroon 1', 
    'Tomato Yellow 1'
])

# --- 2. DATABASE SẢN PHẨM ---
PRODUCT_DB = {
    "Apple Braeburn 1":      {"name": "Táo Braeburn (Nhập khẩu)", "plu": "4101", "price": "70,000"},
    "Apple Crimson Snow 1":  {"name": "Táo Tuyết (Crimson Snow)", "plu": "3015", "price": "120,000"},
    "Apple Golden 1":        {"name": "Táo Vàng (Golden)",        "plu": "4020", "price": "80,000"},
    "Apple Granny Smith 1":  {"name": "Táo Xanh (Granny Smith)",  "plu": "4139", "price": "100,000"},
    "Apple Pink Lady 1":     {"name": "Táo Hồng (Pink Lady)",     "plu": "4130", "price": "110,000"},
    "Apple Red 1":           {"name": "Táo Đỏ Thường",            "plu": "4015", "price": "50,000"},
    "Apple Red Delicious 1": {"name": "Táo Đỏ Mỹ (Delicious)",    "plu": "4016", "price": "85,000"},
    "Apple Red Yellow 1":    {"name": "Táo Lai (Red Yellow)",     "plu": "4173", "price": "75,000"},
    "Pear 1":                {"name": "Lê Thường (Việt Nam)",     "plu": "4409", "price": "40,000"},
    "Pear Abate 1":          {"name": "Lê Abate (Ý)",             "plu": "3012", "price": "95,000"},
    "Pear Forelle 1":        {"name": "Lê Forelle (Nam Phi)",     "plu": "4418", "price": "110,000"},
    "Pear Kaiser 1":         {"name": "Lê Nâu (Kaiser)",          "plu": "3310", "price": "90,000"},
    "Pear Monster 1":        {"name": "Lê Khổng Lồ",              "plu": "3020", "price": "60,000"},
    "Pear Williams 1":       {"name": "Lê Williams (Xanh/Thơm)",  "plu": "4401", "price": "85,000"},
    "Tomato 1":              {"name": "Cà Chua Thường",           "plu": "4064", "price": "25,000"},
    "Tomato Cherry Red 1":   {"name": "Cà Chua Bi Đỏ (Cherry)",   "plu": "4796", "price": "60,000"},
    "Tomato Heart 1":        {"name": "Cà Chua Tim (Beefsteak)",  "plu": "4799", "price": "70,000"},
    "Tomato Maroon 1":       {"name": "Cà Chua Đen (Socola)",     "plu": "4801", "price": "85,000"},
    "Tomato Yellow 1":       {"name": "Cà Chua Vàng",             "plu": "4805", "price": "80,000"},
    "Orange 1":              {"name": "Cam Vàng Navel",           "plu": "3107", "price": "85,000"},
    "Mandarine 1":           {"name": "Quýt Đường",               "plu": "4055", "price": "50,000"},
    "Clementine 1":          {"name": "Quýt Clementine (Ngọt)",   "plu": "4450", "price": "110,000"},
    "Tangelo 1":             {"name": "Cam Lai (Tangelo)",        "plu": "4327", "price": "75,000"},
    "Grapefruit Pink 1":     {"name": "Bưởi Hồng (Ruby)",         "plu": "4285", "price": "75,000"},
    "Grapefruit White 1":    {"name": "Bưởi Năm Roi (Trắng)",     "plu": "4284", "price": "60,000"},
    "Lemon 1":               {"name": "Chanh Vàng (Mỹ)",          "plu": "4053", "price": "120,000"},
    "Limes 1":               {"name": "Chanh Xanh (Không hạt)",   "plu": "4048", "price": "35,000"},
    "Avocado 1":             {"name": "Bơ Sáp 034",               "plu": "4225", "price": "45,000"},
    "Avocado ripe 1":        {"name": "Bơ Chín (Ăn liền)",        "plu": "4046", "price": "55,000"},
    "Mango 1":               {"name": "Xoài Keo (Xanh/Vàng)",     "plu": "4051", "price": "30,000"},
    "Mango Red 1":           {"name": "Xoài Úc (Đỏ)",             "plu": "4959", "price": "70,000"},
    "Papaya 1":              {"name": "Đu Đủ Ruột Đỏ",            "plu": "3111", "price": "35,000"},
    "Pineapple 1":           {"name": "Thơm/Dứa Mật",             "plu": "4430", "price": "20,000"},
    "Pitahaya Red 1":        {"name": "Thanh Long Ruột Đỏ",       "plu": "3040", "price": "40,000"},
    "Passion Fruit 1":       {"name": "Chanh Dây (Mác mác)",      "plu": "4397", "price": "45,000"},
    "Mangostan 1":           {"name": "Măng Cụt Lái Thiêu",       "plu": "3042", "price": "90,000"},
    "Pomegranate 1":         {"name": "Lựu Đỏ Ai Cập",            "plu": "4445", "price": "130,000"},
    "Nectarine 1":           {"name": "Xuân Đào (Trơn)",          "plu": "4036", "price": "250,000"},
    "Peach 1":               {"name": "Đào Lông (Mỹ)",            "plu": "4038", "price": "220,000"},
    "Peach Flat 1":          {"name": "Đào Dẹt (Bánh Rán)",       "plu": "4444", "price": "280,000"},
    "Plum 1":                {"name": "Mận Tím (Plum)",           "plu": "4040", "price": "180,000"},
    "Apricot 1":             {"name": "Mơ Vàng (Apricot)",        "plu": "4218", "price": "200,000"},
    "Pepper Red 1":          {"name": "Ớt Chuông Đỏ",             "plu": "4088", "price": "80,000"},
    "Pepper Green 1":        {"name": "Ớt Chuông Xanh",           "plu": "4065", "price": "50,000"},
    "Pepper Yellow 1":       {"name": "Ớt Chuông Vàng",           "plu": "4689", "price": "80,000"},
    "Potato Red 1":          {"name": "Khoai Tây Hồng (Đà Lạt)",  "plu": "4073", "price": "40,000"},
    "Potato White 1":        {"name": "Khoai Tây Trắng",          "plu": "4083", "price": "30,000"},
    "Ginger Root 1":         {"name": "Gừng Sẻ (Củ)",             "plu": "4612", "price": "50,000"},
    "Cucumber 1":            {"name": "Dưa Leo (Baby)",           "plu": "4062", "price": "25,000"},
    "Banana 1":              {"name": "Chuối Cavendish",          "plu": "4011", "price": "25,000"},
}

# --- 3. LOAD MODEL ---
print(f"🔄 Đang tải mô hình trên thiết bị: {DEVICE}")

try:
    model = Swinv2ForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(class_names),
        ignore_mismatched_sizes=True
    )
    # Load weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Đã nạp thành công weights Swin Transformer V2!")
    else:
        print(f"⚠️ CẢNH BÁO: Không tìm thấy file '{MODEL_PATH}'. Hãy chắc chắn bạn đã để file model cùng thư mục.")
    
    model.to(DEVICE)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

except Exception as e:
    print(f"❌ Lỗi tải model: {e}")

# --- 4. HÀM DỰ ĐOÁN (PREDICT FUNCTION) ---
def predict_smart(image):
    if image is None: return "<h2>⚠️ Vui lòng tải ảnh lên.</h2>"
    
    # Chuyển ảnh sang PIL RGB
    img_pil = Image.fromarray(image).convert("RGB")

    # Test-Time Augmentation (TTA)
    tta_transforms = [
        transforms.Compose([transforms.Resize((192, 192)), transforms.ToTensor(), transforms.Normalize(mean=processor.image_mean, std=processor.image_std)]),
        transforms.Compose([transforms.Resize((192, 192)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(mean=processor.image_mean, std=processor.image_std)]),
    ]

    logits_list = []
    with torch.no_grad():
        for t in tta_transforms:
            input_tensor = t(img_pil).unsqueeze(0).to(DEVICE)
            output = model(input_tensor).logits
            logits_list.append(output)
    
    # Cộng gộp kết quả
    avg_logits = torch.stack(logits_list).mean(dim=0)
    temperature = 0.5 
    final_probs = F.softmax(avg_logits / temperature, dim=1)
    
    # Lấy Top 1
    confidence, pred_idx = torch.max(final_probs[0], 0)
    pred_idx = pred_idx.item()
    confidence = confidence.item()
    
    pred_label = class_names[pred_idx]
    info = PRODUCT_DB.get(pred_label, {"name": pred_label, "plu": "---", "price": "Liên hệ"})

    # Tạo HTML hiển thị
    top3_prob, top3_idx = torch.topk(final_probs[0], 3)
    top3_html = ""
    for i in range(3):
        idx = top3_idx[i].item()
        score = top3_prob[i].item() * 100
        name = class_names[idx]
        short_name = PRODUCT_DB.get(name, {}).get('name', name).split('(')[0]
        bar_color = "#28a745" if i == 0 else "#6c757d"
        font_weight = "bold" if i == 0 else "normal"
        top3_html += f"""
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
                <span style="font-weight: {font_weight};">{short_name}</span>
                <span>{score:.1f}%</span>
            </div>
            <div style="width: 100%; background-color: #e9ecef; border-radius: 4px; height: 6px;">
                <div style="background-color: {bar_color}; height: 6px; border-radius: 4px; width: {score}%;"></div>
            </div>
        </div>
        """

    html_content = f"""
    <div style="font-family: 'Segoe UI', sans-serif; max-width: 400px; margin: auto; border-radius: 15px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); background-color: #ffffff;">
        <div style="background: linear-gradient(135deg, #0061f2 0%, #00c6f9 100%); padding: 15px; text-align: center; color: white;">
            <h2 style="margin: 0; font-size: 18px;">🛒 KẾT QUẢ QUÉT AI</h2>
        </div>
        <div style="padding: 20px;">
            <div style="text-align: center;">
                <span style="background-color: #e3f2fd; color: #0d47a1; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: bold;">
                    ĐỘ TIN CẬY: {confidence*100:.1f}%
                </span>
            </div>
            <h1 style="margin: 10px 0 5px 0; font-size: 22px; color: #333; text-align: center; line-height: 1.3;">
                {info['name']}
            </h1>
            <div style="display: flex; justify-content: space-between; align-items: center; margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 10px; border: 1px dashed #ced4da;">
                <div style="text-align: center; flex: 1; border-right: 1px solid #ddd;">
                    <div style="font-size: 11px; color: #666; text-transform: uppercase;">MÃ PLU</div>
                    <div style="font-size: 24px; font-weight: 900; color: #dc3545;">{info['plu']}</div>
                </div>
                <div style="text-align: center; flex: 1;">
                    <div style="font-size: 11px; color: #666; text-transform: uppercase;">ĐƠN GIÁ</div>
                    <div style="font-size: 20px; font-weight: bold; color: #28a745;">{info['price']}</div>
                    <div style="font-size: 10px; color: #999;">/ kg</div>
                </div>
            </div>
            <div style="border-top: 1px solid #eee; padding-top: 15px;">
                <div style="font-size: 11px; color: #888; margin-bottom: 10px; text-transform: uppercase; font-weight: bold;">
                    PHÂN TÍCH XÁC SUẤT:
                </div>
                {top3_html}
            </div>
        </div>
    </div>
    """
    return html_content

# --- 5. GIAO DIỆN GRADIO ---
with gr.Blocks(title="Trợ Lý Thu Ngân AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #047857; margin-bottom: 0;">🥦 TRỢ LÝ THU NGÂN AI 🍎</h1>
        <p style="color: #64748b;">Hệ thống nhận diện Nông sản tự động - CS231</p>
    </div>
    """)
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", height=400, label="📸 Ảnh sản phẩm")
            btn = gr.Button("🔍 QUÉT NGAY", variant="primary")
        with gr.Column():
            output = gr.HTML(label="Kết quả")
    
    btn.click(fn=predict_smart, inputs=input_img, outputs=output)

if __name__ == "__main__":
    demo.launch()