# exhibit_api_fixed.py
import os
import cv2
import numpy as np
import chromadb
from PIL import Image
from flask import Flask, request, jsonify
import logging
import io
import requests  # 添加requests库用于下载图片
import time
import uuid
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

print("🚀 启动修复版艺术展品识别API...")

# 连接向量数据库
client = chromadb.PersistentClient(path="./exhibit_vector_db")
collection = client.get_collection("art_exhibits")

print(f"✅ 数据库连接成功! 包含 {collection.count()} 条记录")


class RobustFeatureExtractor:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=300)
        logger.info("特征提取器初始化完成")

    def extract_orb_features(self, image):
        """使用ORB提取特征 - 与构建数据库时一致"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            keypoints, descriptors = self.orb.detectAndCompute(gray, None)

            if descriptors is not None and len(descriptors) > 5:
                mean_desc = descriptors.mean(axis=0)
                if len(mean_desc) > 32:
                    mean_desc = mean_desc[:32]
                elif len(mean_desc) < 32:
                    mean_desc = np.pad(mean_desc, (0, 32 - len(mean_desc)))

                norm = np.linalg.norm(mean_desc)
                return mean_desc / norm if norm > 0 else mean_desc
            return None
        except Exception as e:
            logger.warning(f"ORB特征提取失败: {e}")
            return None

    def extract_color_features(self, image):
        """提取颜色特征 - 与构建数据库时一致"""
        try:
            if len(image.shape) == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
                hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
                hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])

                hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
                hist = cv2.normalize(hist, hist).flatten()
                return hist
            else:
                hist = cv2.calcHist([image], [0], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                return hist
        except Exception as e:
            logger.warning(f"颜色特征提取失败: {e}")
            return None

    def extract_texture_features(self, image):
        """提取纹理特征 - 与构建数据库时一致"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # 计算LBP纹理特征
            lbp = self.local_binary_pattern(gray)
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=[0, 256])
            hist = hist.astype(np.float32)
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        except Exception as e:
            logger.warning(f"纹理特征提取失败: {e}")
            return None

    def local_binary_pattern(self, image, P=8, R=1):
        """计算局部二值模式"""
        height, width = image.shape
        lbp = np.zeros((height - 2, width - 2), dtype=np.uint8)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                center = image[i, j]
                code = 0
                code |= (image[i - 1, j - 1] > center) << 7
                code |= (image[i - 1, j] > center) << 6
                code |= (image[i - 1, j + 1] > center) << 5
                code |= (image[i, j + 1] > center) << 4
                code |= (image[i + 1, j + 1] > center) << 3
                code |= (image[i + 1, j] > center) << 2
                code |= (image[i + 1, j - 1] > center) << 1
                code |= (image[i, j - 1] > center) << 0
                lbp[i - 1, j - 1] = code
        return lbp

    def extract_features(self, image_path):
        """综合特征提取 - 与构建数据库时完全一致"""
        try:
            # 使用PIL读取图片（支持中文路径）
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # 调整图片大小
            max_size = 800
            if max(pil_image.size) > max_size:
                ratio = max_size / max(pil_image.size)
                new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

            # 转换为OpenCV格式
            cv2_image = self.pil_to_cv2(pil_image)

            features = []

            # 方法1: ORB特征
            orb_feat = self.extract_orb_features(cv2_image)
            if orb_feat is not None:
                features.extend(orb_feat)
            else:
                features.extend([0] * 32)

            # 方法2: 颜色特征
            color_feat = self.extract_color_features(cv2_image)
            if color_feat is not None:
                features.extend(color_feat)
            else:
                features.extend([0] * 24)

            # 方法3: 纹理特征
            texture_feat = self.extract_texture_features(cv2_image)
            if texture_feat is not None:
                features.extend(texture_feat)
            else:
                features.extend([0] * 256)

            # 确保特征向量长度一致
            target_length = 312
            if len(features) > target_length:
                features = features[:target_length]
            elif len(features) < target_length:
                features.extend([0] * (target_length - len(features)))

            feature_vector = np.array(features, dtype=np.float32)
            norm = np.linalg.norm(feature_vector)

            if norm > 0:
                feature_vector = feature_vector / norm

            return feature_vector.tolist()

        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return None

    def pil_to_cv2(self, pil_image):
        """PIL图像转OpenCV格式"""
        cv2_image = np.array(pil_image)
        cv2_image = cv2_image[:, :, ::-1].copy()
        return cv2_image


# 全局特征提取器
feature_extractor = RobustFeatureExtractor()


def extract_query_features(image_bytes):
    """查询特征提取 - 与构建时完全一致"""
    try:
        # 使用PIL读取图片字节
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # 调整大小
        max_size = 800
        if max(pil_image.size) > max_size:
            ratio = max_size / max(pil_image.size)
            new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

        # 保存临时文件用于特征提取
        temp_path = "temp_query_image.jpg"
        pil_image.save(temp_path)

        # 使用与构建时完全相同的方法
        features = feature_extractor.extract_features(temp_path)

        # 删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return features

    except Exception as e:
        logger.error(f"查询特征提取失败: {e}")
        return None


def identify_from_image_data(image_data):
    """通用的识别逻辑"""
    # 提取特征（使用与构建时完全相同的方法）
    query_vector = extract_query_features(image_data)

    if query_vector is None:
        return {"status": "error", "message": "特征提取失败"}

    # 搜索相似展品
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=5,
        include=["metadatas", "distances"]
    )

    if results and results['metadatas']:
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        # 将距离转换为相似度分数
        similarities = [1 - distance for distance in distances]

        # 找到相似度最高的结果
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        best_metadata = metadatas[best_match_idx]

        # 降低阈值到0.3
        if best_similarity >= 0.3:
            return {
                "status": "success",
                "exhibit_id": best_metadata["exhibit_id"],
                "confidence": round(best_similarity, 4),
                "category": best_metadata["category"],
                "message": f"识别成功: {best_metadata['exhibit_id']}",
                "all_matches": [
                    {
                        "exhibit_id": meta["exhibit_id"],
                        "confidence": round(sim, 4)
                    }
                    for meta, sim in zip(metadatas, similarities)
                ]
            }

    return {
        "status": "not_found",
        "exhibit_id": None,
        "confidence": 0.0,
        "message": "未找到匹配的展品"
    }


@app.route('/identify', methods=['POST'])
def identify_exhibit():
    """识别展品 - 文件上传方式"""
    try:
        logger.info("收到文件识别请求")

        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "没有上传文件"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "没有选择文件"}), 400

        # 读取图片数据
        image_data = file.read()
        if len(image_data) == 0:
            return jsonify({"status": "error", "message": "上传的文件为空"}), 400

        # 使用通用识别逻辑
        result = identify_from_image_data(image_data)
        return jsonify(result)

    except Exception as e:
        logger.error(f"文件识别过程出错: {e}")
        return jsonify({"status": "error", "message": f"识别过程出错: {str(e)}"}), 500


# === 修改 identify_by_url 函数 ===
@app.route('/identify_by_url', methods=['POST'])
def identify_by_url():
    try:
        data = request.get_json()
        image_url = data.get('image_url')

        # 生成唯一请求标识
        request_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%H:%M:%S")

        logger.info(f"🆕 请求 {request_id} | 时间 {timestamp} | 开始识别: {image_url[:50]}...")

        # 下载图片
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(image_url, headers=headers, timeout=30)

        if response.status_code != 200:
            logger.error(f"❌ 请求 {request_id} | 下载失败: HTTP {response.status_code}")
            return jsonify({
                "status": "error",
                "message": f"下载图片失败: HTTP {response.status_code}",
                "request_id": request_id
            }), 400

        # 直接使用 identify_from_image_data 函数（你实际使用的函数）
        result = identify_from_image_data(response.content)

        # 如果上面报错，尝试使用 extract_query_features
        # result = process_image_recognition(response.content, request_id)

        # 增强返回结果
        if result['status'] == 'success':
            result.update({
                'request_id': request_id,
                'timestamp': timestamp,
                'is_new_request': True,
                'cache_used': False,
                'message': f"🆕 识别成功: {result['exhibit_id']} (请求ID: {request_id})"
            })
            logger.info(f"✅ 请求 {request_id} | 识别成功: {result['exhibit_id']}")
        else:
            result.update({
                'request_id': request_id,
                'timestamp': timestamp,
                'is_new_request': True
            })
            logger.info(f"❌ 请求 {request_id} | 识别失败: {result.get('message', '未知错误')}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"💥 请求处理异常: {e}")
        return jsonify({
            "status": "error",
            "message": f"识别过程出错: {str(e)}",
            "request_id": request_id if 'request_id' in locals() else 'unknown'
        }), 500


# === 新增：统一的图片识别处理函数 ===
def process_image_recognition(image_data, request_id):
    """统一的图片识别处理"""
    try:
        # 提取特征
        query_vector = extract_query_features(image_data)

        if query_vector is None:
            return {"status": "error", "message": "特征提取失败"}

        # 搜索相似展品
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=5,
            include=["metadatas", "distances"]
        )

        if results and results['metadatas']:
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]

            # 将距离转换为相似度分数
            similarities = [1 - distance for distance in distances]

            # 找到相似度最高的结果
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            best_metadata = metadatas[best_match_idx]

            # 降低阈值到0.3
            if best_similarity >= 0.3:
                return {
                    "status": "success",
                    "exhibit_id": best_metadata["exhibit_id"],
                    "confidence": round(best_similarity, 4),
                    "category": best_metadata.get("category", ""),
                    "similarities": similarities,  # 返回所有相似度用于调试
                    "all_matches": [
                        {
                            "exhibit_id": meta["exhibit_id"],
                            "confidence": round(sim, 4)
                        }
                        for meta, sim in zip(metadatas, similarities)
                    ]
                }

        return {
            "status": "not_found",
            "exhibit_id": None,
            "confidence": 0.0,
            "message": "未找到匹配的展品"
        }

    except Exception as e:
        logger.error(f"特征识别过程出错: {e}")
        return {"status": "error", "message": f"识别过程出错: {str(e)}"}


@app.route('/debug_clear_cache', methods=['POST'])
def debug_clear_cache():
    """清理可能的缓存"""
    try:
        # 如果有任何全局缓存变量，在这里清理
        global feature_cache
        if 'feature_cache' in globals():
            feature_cache.clear()

        # 清理可能的函数缓存
        import functools
        if hasattr(extract_query_features, 'cache'):
            extract_query_features.cache_clear()

        return jsonify({
            "status": "success",
            "message": "缓存已清理",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"清理缓存失败: {str(e)}"
        })


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "exhibit-recognition",
        "database_records": collection.count()
    })


@app.route('/exhibits', methods=['GET'])
def list_exhibits():
    try:
        results = collection.get(limit=1000)
        exhibits = {}
        for metadata in results['metadatas']:
            exhibit_id = metadata['exhibit_id']
            if exhibit_id not in exhibits:
                exhibits[exhibit_id] = {
                    "category": metadata['category'],
                    "image_count": 0
                }
            exhibits[exhibit_id]['image_count'] += 1

        return jsonify({
            "status": "success",
            "total_exhibits": len(exhibits),
            "exhibits": exhibits
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"获取展品列表失败: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)