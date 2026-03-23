from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import torch
import os
import sys
import types
import io
import base64
import time

# --- إصلاح التوافقية ---
import torchvision.transforms.functional as F
if 'torchvision.transforms.functional_tensor' not in sys.modules:
    ft_module = types.ModuleType('torchvision.transforms.functional_tensor')
    for attr_name in dir(F):
        if not attr_name.startswith('_'):
            setattr(ft_module, attr_name, getattr(F, attr_name))
    sys.modules['torchvision.transforms.functional_tensor'] = ft_module

# --- تحميل النموذج (يتم مرة واحدة عند بدء التشغيل) ---
print("⏳ تحميل نموذج الذكاء الاصطناعي...")
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model_path = 'weights/RealESRGAN_x4plus.pth'

# تحميل الوزن إذا لم يكن موجوداً (للتشغيل المحلي أو السحابي)
if not os.path.exists(model_path):
    os.makedirs('weights', exist_ok=True)
    print("Downloading model weights...")
    import requests
    url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    r = requests.get(url)
    with open(model_path, 'wb') as f:
        f.write(r.content)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=False, device=device)
print(f"✅ النموذج جاهز على {device}!")

app = Flask(__name__)

# --- المسارات (Routes) ---

@app.route('/')
def index():
    # صفحة الهاتف (واجهة المستخدم)
    return render_template('index.html')

@app.route('/process_network', methods=['POST'])
def process_network():
    """
    هذا المسار يستقبل الصورة من الهاتف، يوزعها، ويعالجها.
    في هذه التجربة، السيرفر يقوم بدور "الحاسوب القوي" ويعالج الجزء الأصعب.
    """
    try:
        # 1. استقبال الصورة
        file = request.files['image']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        h, w = img.shape[:2]
        mid = w // 2
        
        # 2. التقطيع (Simulating Sharding)
        left_tile = img[:, :mid]   # الجزء الأيسر
        right_tile = img[:, mid:]  # الجزء الأيمن

        # 3. المعالجة (التوزيع)
        
        # أ. الهاتف (تمت محاكاته هنا بالفلتر البسيط للسرعة، أو يمكنك إرسال هذا الجزء للهاتف لمعالجته لو كان الطلب AJAX معقداً)
        # للتبسيط: سنقوم بعمل "فلتر" بسيط هنا يمثل معالجة الهاتف
        print("⚙️ معالجة جزء (محاكاة الهاتف)...")
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        left_processed = cv2.filter2D(left_tile, -1, kernel)
        # تكبيره ليتناسب مع النتيجة الأخرى
        left_processed = cv2.resize(left_processed, (mid*4, h*4))

        # ب. السيرفر (GPU Power)
        print("⚙️ معالجة جزء (قوة الحاسوب/GPU)...")
        right_processed, _ = upsampler.enhance(right_tile, outscale=4)

        # 4. التجميع (Stitching)
        print("🧩 تجميع النتيجة...")
        final_image = np.hstack((left_processed, right_processed))
        
        # 5. إرسال النتيجة كصورة
        _, buffer = cv2.imencode('.jpg', final_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

    except Exception as e:
        print(f"❌ خطأ: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # للتشغيل المحلي
    app.run(host='0.0.0.0', port=5000, debug=True)