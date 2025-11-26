# build_vector_db_fixed.py
import os
import cv2
import numpy as np
import chromadb
from PIL import Image
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RobustFeatureExtractor:
    def __init__(self):
        """使用多种特征提取方法确保稳定性"""
        # 方法1: ORB特征
        self.orb = cv2.ORB_create(nfeatures=300)
        logger.info("离线特征提取器初始化完成")

    def pil_to_cv2(self, pil_image):
        """将PIL图像转换为OpenCV格式"""
        # PIL是RGB, OpenCV是BGR
        cv2_image = np.array(pil_image)
        cv2_image = cv2_image[:, :, ::-1].copy()  # RGB to BGR
        return cv2_image

    def extract_orb_features(self, image):
        """使用ORB提取特征"""
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            keypoints, descriptors = self.orb.detectAndCompute(gray, None)

            if descriptors is not None and len(descriptors) > 5:
                # 使用描述符的均值作为特征向量
                mean_desc = descriptors.mean(axis=0)

                # 确保向量长度为32（ORB默认）
                if len(mean_desc) > 32:
                    mean_desc = mean_desc[:32]
                elif len(mean_desc) < 32:
                    mean_desc = np.pad(mean_desc, (0, 32 - len(mean_desc)))

                # 归一化
                norm = np.linalg.norm(mean_desc)
                return mean_desc / norm if norm > 0 else mean_desc
            return None
        except Exception as e:
            logger.warning(f"ORB特征提取失败: {e}")
            return None

    def extract_color_features(self, image):
        """使用颜色直方图作为特征"""
        try:
            if len(image.shape) == 3:
                # 计算HSV颜色直方图
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
                hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
                hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])

                # 合并并归一化
                hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
                hist = cv2.normalize(hist, hist).flatten()
                return hist
            else:
                # 灰度图，只计算亮度直方图
                hist = cv2.calcHist([image], [0], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                return hist
        except Exception as e:
            logger.warning(f"颜色特征提取失败: {e}")
            return None

    def extract_texture_features(self, image):
        """使用纹理特征"""
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
        """综合特征提取 - 使用PIL读取图片避免中文路径问题"""
        try:
            # 使用PIL读取图片（支持中文路径）
            pil_image = Image.open(image_path)

            # 转换为RGB（确保3通道）
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # 调整图片大小以加快处理速度
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
                # 如果ORB失败，用零填充
                features.extend([0] * 32)

            # 方法2: 颜色特征
            color_feat = self.extract_color_features(cv2_image)
            if color_feat is not None:
                features.extend(color_feat)
            else:
                features.extend([0] * 24)  # 8+8+8=24

            # 方法3: 纹理特征
            texture_feat = self.extract_texture_features(cv2_image)
            if texture_feat is not None:
                features.extend(texture_feat)
            else:
                features.extend([0] * 256)

            # 确保特征向量长度一致
            target_length = 312  # 32 + 24 + 256
            if len(features) > target_length:
                features = features[:target_length]
            elif len(features) < target_length:
                features.extend([0] * (target_length - len(features)))

            feature_vector = np.array(features, dtype=np.float32)
            norm = np.linalg.norm(feature_vector)

            if norm > 0:
                feature_vector = feature_vector / norm

            logger.info(f"✅ 成功提取特征: {os.path.basename(image_path)}")
            return feature_vector.tolist()

        except Exception as e:
            logger.error(f"❌ 特征提取失败 {os.path.basename(image_path)}: {e}")
            return None


class ExhibitVectorizer:
    def __init__(self):
        self.feature_extractor = RobustFeatureExtractor()
        self.client = chromadb.PersistentClient(path="./exhibit_vector_db")
        logger.info("展品向量化器初始化完成")

    def create_collection(self, collection_name="art_exhibits"):
        """创建向量数据库集合"""
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"向量数据库集合 '{collection_name}' 创建成功")

    def scan_dataset(self, dataset_path):
        """扫描数据集并整理展品结构"""
        exhibits = {}

        logger.info(f"开始扫描数据集: {dataset_path}")

        if not os.path.exists(dataset_path):
            logger.error(f"数据集路径不存在: {dataset_path}")
            return exhibits

        # 遍历分类文件夹
        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            if not os.path.isdir(category_path):
                continue

            logger.info(f"扫描分类: {category}")

            # 遍历分类下的所有项目
            for item in os.listdir(category_path):
                item_path = os.path.join(category_path, item)

                if os.path.isfile(item_path) and item.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # 单个图片文件
                    exhibit_id = f"{category}_{os.path.splitext(item)[0]}"
                    if exhibit_id not in exhibits:
                        exhibits[exhibit_id] = []
                    exhibits[exhibit_id].append(item_path)

                elif os.path.isdir(item_path):
                    # 图片文件夹
                    exhibit_id = f"{category}_{item}"
                    image_list = []

                    for img_file in os.listdir(item_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(item_path, img_file)
                            image_list.append(img_path)

                    if image_list:
                        exhibits[exhibit_id] = image_list

        total_images = sum(len(images) for images in exhibits.values())
        logger.info(f"数据集扫描完成: 共 {len(exhibits)} 个展品, {total_images} 张图片")

        return exhibits

    def build_vector_database(self, dataset_path):
        """构建向量数据库"""
        # 扫描数据集
        exhibits = self.scan_dataset(dataset_path)

        if not exhibits:
            logger.error("没有找到任何展品图片，请检查数据集路径")
            return False

        # 处理图片并存入向量数据库
        all_embeddings = []
        all_metadatas = []
        all_ids = []

        total_images = sum(len(images) for images in exhibits.values())
        processed_count = 0
        success_count = 0

        logger.info("开始提取特征并构建向量数据库...")

        for exhibit_id, image_list in exhibits.items():
            logger.info(f"处理展品: {exhibit_id} ({len(image_list)} 张图片)")

            for i, image_path in enumerate(image_list):
                processed_count += 1

                print(f"进度: {processed_count}/{total_images} - {os.path.basename(image_path)}")

                # 提取特征
                vector = self.feature_extractor.extract_features(image_path)
                if vector is not None:
                    all_embeddings.append(vector)
                    all_metadatas.append({
                        "exhibit_id": exhibit_id,
                        "image_path": image_path,
                        "category": exhibit_id.split('_')[0],
                        "image_name": os.path.basename(image_path),
                        "angle": f"angle_{i}",
                        "type": "exhibit_reference"
                    })
                    all_ids.append(f"{exhibit_id}_{i}")
                    success_count += 1

        # 保存到向量数据库
        if all_embeddings:
            self.collection.add(
                embeddings=all_embeddings,
                metadatas=all_metadatas,
                ids=all_ids
            )

            logger.info(f"✅ 向量数据库构建成功!")
            logger.info(f"   成功处理: {success_count}/{total_images} 张图片")
            logger.info(f"   展品数量: {len(exhibits)} 个")
            logger.info(f"   向量维度: {len(all_embeddings[0])} 维")

            # 显示展品列表
            print("\n📋 已处理的展品列表:")
            for exhibit_id in sorted(exhibits.keys()):
                print(f"   🎨 {exhibit_id}")

            return True
        else:
            logger.error("❌ 没有成功提取任何特征向量")
            return False


def main():
    """主函数"""
    print("=" * 60)
    print("🎯 艺术展品识别系统 - 修复版（支持中文路径）")
    print("=" * 60)

    # 检查依赖
    try:
        import cv2
        import chromadb
        from PIL import Image
        print("✅ 依赖检查通过")
    except ImportError as e:
        print(f"❌ 依赖缺失: {e}")
        print("请运行: pip install opencv-python chromadb pillow numpy")
        return

    # 数据集路径
    dataset_path = r"D:\大三上学期\人形机器人项目实践\原始图片\原始图片\产品设计"

    if not os.path.exists(dataset_path):
        print(f"❌ 数据集路径不存在: {dataset_path}")
        print("请检查路径是否正确")
        return

    print(f"📁 数据集路径: {dataset_path}")

    # 构建向量数据库
    vectorizer = ExhibitVectorizer()
    vectorizer.create_collection()

    success = vectorizer.build_vector_database(dataset_path)

    if success:
        print("\n🎉 恭喜！向量数据库构建完成！")
        print("下一步: 运行 API 服务来测试识别效果")
    else:
        print("\n💥 构建失败，请检查以上错误信息")


if __name__ == "__main__":
    main()