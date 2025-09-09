import os
import shapefile
from shapely.geometry import shape, Point, LineString, Polygon as ShapelyPolygon
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from typing import Dict, List, Tuple, Any
import random

# ───────────────── Font & Display ───────────────── #
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------
# 路径设置
ROOT = r"C:\Users\yh\Desktop\Data2\Data"  # 数据根目录
OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ROOT,
    "output"
)
os.makedirs(OUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 通用工具
def first_shp(path):
    """返回目录中第一个 .shp 完整路径;若无返回 None"""
    if not os.path.isdir(path):
        return None
    for f in os.listdir(path):
        if f.lower().endswith(".shp"):
            return os.path.join(path, f)
    return None


def build_geom_list(shp_path):
    """读取 shapefile → [shapely.geometry, ...]"""
    if not shp_path or not os.path.exists(shp_path):
        return []
    rdr = shapefile.Reader(shp_path)
    return [shape(shp.__geo_interface__) for shp in rdr.shapes()]


def build_geom_dict(shp_path):
    """读取 shapefile → {name: shapely.geometry}"""
    if not shp_path or not os.path.exists(shp_path):
        return {}

    rdr = shapefile.Reader(shp_path)
    field_names = [f[0] for f in rdr.fields[1:]]
    name_f = next((f for f in field_names if "name" in f.lower()), field_names[0] if field_names else None)

    if not name_f:
        # 如果没有name字段，使用索引
        d = {}
        for i, shp in enumerate(rdr.shapes()):
            d[f"feature_{i}"] = shape(shp.__geo_interface__)
        return d

    d = {}
    for i, (rec, shp) in enumerate(zip(rdr.records(), rdr.shapes())):
        try:
            raw = rec[field_names.index(name_f)]
            nm = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else str(raw)
            d[nm.strip()] = shape(shp.__geo_interface__)
        except:
            d[f"feature_{i}"] = shape(shp.__geo_interface__)
    return d


class MapDataLoader:
    """地图数据加载器，负责读取shapefile和关系数据"""

    def __init__(self, root_path: str):
        self.root_path = root_path

    def load_group_data(self, group_name: str) -> Dict[str, Any]:
        """加载一个组的所有数据"""
        group_path = os.path.join(self.root_path, group_name)

        area_list, line_list, point_list = [], [], []
        area_dict, line_dict, point_dict = {}, {}, {}
        rel_frames = []

        # 遍历所有Sample
        for sample in os.listdir(group_path):
            spath = os.path.join(group_path, sample)
            if not os.path.isdir(spath):
                continue

            # 读取几何数据
            area_shp = first_shp(os.path.join(spath, "my_zip_area"))
            line_shp = first_shp(os.path.join(spath, "my_zip_line"))
            point_shp = first_shp(os.path.join(spath, "my_zip_point"))
            rel_xlsx = os.path.join(spath, "relations.xlsx")

            if area_shp:
                area_list += build_geom_list(area_shp)
                area_dict.update(build_geom_dict(area_shp))
            if line_shp:
                line_list += build_geom_list(line_shp)
                line_dict.update(build_geom_dict(line_shp))
            if point_shp:
                point_list += build_geom_list(point_shp)
                point_dict.update(build_geom_dict(point_shp))
            if os.path.exists(rel_xlsx):
                try:
                    rel_frames.append(pd.read_excel(rel_xlsx))
                except:
                    pass

        # 合并关系数据
        relations = pd.concat(rel_frames, ignore_index=True) if rel_frames else pd.DataFrame()

        return {
            'areas': area_dict,
            'lines': line_dict,
            'points': point_dict,
            'areas_list': area_list,
            'lines_list': line_list,
            'points_list': point_list,
            'relations': relations,
            'group_path': group_path
        }

    def convert_to_standard_format(self, group_data: Dict[str, Any]) -> Tuple[List, List, List, List]:
        """转换为标准格式（点、线、面列表）"""
        points = []
        lines = []
        polygons = []
        relationships = []

        # 名称到索引的映射
        name_to_idx = {}

        # 处理点（使用列表形式）
        for i, geom in enumerate(group_data['points_list']):
            if geom.geom_type == 'Point':
                points.append((geom.x, geom.y))
                # 为列表中的每个点创建映射
                for name, dict_geom in group_data['points'].items():
                    if dict_geom.equals(geom):
                        name_to_idx[name] = ('point', len(points) - 1)
                        break

        # 处理线（使用列表形式）
        for i, geom in enumerate(group_data['lines_list']):
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                lines.append(coords)
                # 为列表中的每条线创建映射
                for name, dict_geom in group_data['lines'].items():
                    if dict_geom.equals(geom):
                        name_to_idx[name] = ('line', len(lines) - 1)
                        break

        # 处理面（使用列表形式）
        for i, geom in enumerate(group_data['areas_list']):
            if geom.geom_type == 'Polygon':
                coords = list(geom.exterior.coords)
                polygons.append(coords)
                # 为列表中的每个面创建映射
                for name, dict_geom in group_data['areas'].items():
                    if dict_geom.equals(geom):
                        name_to_idx[name] = ('polygon', len(polygons) - 1)
                        break

        # 处理关系
        if not group_data['relations'].empty:
            for _, row in group_data['relations'].iterrows():
                src_name = str(row.get('src_name', '')).strip()
                dst_name = str(row.get('dst_name', '')).strip()
                rel_type = str(row.get('rel_type', 'Connection')).strip()

                if src_name in name_to_idx and dst_name in name_to_idx:
                    relationships.append({
                        'source': name_to_idx[src_name],
                        'target': name_to_idx[dst_name],
                        'type': rel_type
                    })

        return points, lines, polygons, relationships


class EnhancedSpatialRelationBuilder:
    """增强的空间关系图构建器，包含自定义关系"""

    def __init__(self, distance_threshold: float = 2.0):
        self.distance_threshold = distance_threshold

    def build_enhanced_spatial_graph(self, points: List, lines: List, polygons: List,
                                     relationships: List) -> nx.Graph:
        """构建包含自定义关系的空间关系图"""
        graph = nx.Graph()

        # 添加节点
        node_id = 0
        node_mapping = {}

        # 添加点节点
        for i, p in enumerate(points):
            node_name = f'p{i}'
            graph.add_node(node_name, type='point', geometry=p, coords=p, node_id=node_id)
            node_mapping[('point', i)] = node_name
            node_id += 1

        # 添加线节点
        for i, l in enumerate(lines):
            if len(l) > 1:
                length = sum(np.linalg.norm(np.array(l[j]) - np.array(l[j + 1]))
                             for j in range(len(l) - 1))
                centroid = self.compute_centroid(l)
                node_name = f'l{i}'
                graph.add_node(node_name, type='line', geometry=l, coords=centroid,
                               length=length, node_id=node_id)
                node_mapping[('line', i)] = node_name
                node_id += 1

        # 添加面节点
        for i, poly in enumerate(polygons):
            if len(poly) > 2:
                shapely_poly = ShapelyPolygon(poly)
                area = shapely_poly.area
                perimeter = shapely_poly.length
                centroid = self.compute_centroid(poly)
                node_name = f'poly{i}'
                graph.add_node(node_name, type='polygon', geometry=poly, coords=centroid,
                               area=area, perimeter=perimeter, node_id=node_id)
                node_mapping[('polygon', i)] = node_name
                node_id += 1

        # 添加自定义关系（Gaze, Navigation, Connection）
        relation_type_mapping = {
            'Gaze': 1,
            'Navigation': 2,
            'Connection': 3
        }

        for rel in relationships:
            src_key = rel['source']
            dst_key = rel['target']
            rel_type = rel['type']

            if src_key in node_mapping and dst_key in node_mapping:
                src_node = node_mapping[src_key]
                dst_node = node_mapping[dst_key]

                graph.add_edge(src_node, dst_node,
                               relation_type='custom',
                               custom_type=rel_type,
                               custom_type_id=relation_type_mapping.get(rel_type, 0))

        return graph

    def compute_centroid(self, points: List) -> Tuple[float, float]:
        """计算点集的质心"""
        if not points:
            return (0, 0)
        x_sum = sum(p[0] for p in points)
        y_sum = sum(p[1] for p in points)
        return (x_sum / len(points), y_sum / len(points))


class MapFeatureExtractor:
    """地图特征提取器"""

    def __init__(self):
        self.feature_dim = 128

    def extract_node_features(self, graph: nx.Graph) -> np.ndarray:
        """提取节点特征"""
        if graph.number_of_nodes() == 0:
            return np.array([[0] * 8], dtype=np.float32)

        features = []

        for node, data in graph.nodes(data=True):
            if data['type'] == 'point':
                feat = [
                    data['coords'][0],
                    data['coords'][1],
                    graph.degree(node),
                    self._get_neighbor_type_distribution(graph, node, 'line'),
                    self._get_neighbor_type_distribution(graph, node, 'polygon'),
                    0, 0, 0
                ]
            elif data['type'] == 'line':
                feat = [
                    data['coords'][0],
                    data['coords'][1],
                    data.get('length', 0),
                    graph.degree(node),
                    self._get_neighbor_type_distribution(graph, node, 'point'),
                    self._get_neighbor_type_distribution(graph, node, 'polygon'),
                    0, 0
                ]
            elif data['type'] == 'polygon':
                feat = [
                    data['coords'][0],
                    data['coords'][1],
                    data.get('area', 0),
                    data.get('perimeter', 0),
                    graph.degree(node),
                    self._get_neighbor_type_distribution(graph, node, 'point'),
                    self._get_neighbor_type_distribution(graph, node, 'line'),
                    0
                ]
            else:
                feat = [0] * 8

            features.append(feat[:8])

        return np.array(features, dtype=np.float32)

    def _get_neighbor_type_distribution(self, graph: nx.Graph, node: str,
                                        target_type: str) -> float:
        """计算特定类型邻居的比例"""
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            return 0

        type_count = sum(1 for n in neighbors
                         if graph.nodes[n]['type'] == target_type)
        return type_count / len(neighbors)

    def extract_edge_features(self, graph: nx.Graph) -> np.ndarray:
        """提取边特征"""
        if graph.number_of_edges() == 0:
            return np.array([[0] * 7], dtype=np.float32)

        edge_features = []

        for u, v, data in graph.edges(data=True):
            if data.get('relation_type') == 'custom':
                feat = [
                    data.get('custom_type_id', 0),
                    1.0,
                    0, 0, 0, 0, 0
                ]
            else:
                feat = [
                    0,
                    data.get('distance', 0),
                    data.get('inside_ratio', 0),
                    data.get('iou', 0),
                    0, 0, 0
                ]

            edge_features.append(feat[:7])

        return np.array(edge_features, dtype=np.float32)

    def extract_global_features(self, graph: nx.Graph, points: List,
                                lines: List, polygons: List) -> Dict[str, float]:
        """提取全局特征"""
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        # 关系类型统计
        custom_edges = [e for e in graph.edges(data=True)
                        if e[2].get('relation_type') == 'custom']

        gaze_count = sum(1 for e in custom_edges
                         if e[2].get('custom_type') == 'Gaze')
        nav_count = sum(1 for e in custom_edges
                        if e[2].get('custom_type') == 'Navigation')
        conn_count = sum(1 for e in custom_edges
                         if e[2].get('custom_type') == 'Connection')

        # 几何统计
        total_line_length = 0
        for l in lines:
            if len(l) > 1:
                total_line_length += sum(np.linalg.norm(np.array(l[j]) - np.array(l[j + 1]))
                                         for j in range(len(l) - 1))

        total_polygon_area = 0
        for p in polygons:
            if len(p) > 2:
                total_polygon_area += ShapelyPolygon(p).area

        return {
            'num_points': len(points),
            'num_lines': len(lines),
            'num_polygons': len(polygons),
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': num_edges / (num_nodes + 1),
            'gaze_relations': gaze_count,
            'navigation_relations': nav_count,
            'connection_relations': conn_count,
            'total_line_length': total_line_length,
            'total_polygon_area': total_polygon_area,
            'avg_degree': sum(d for _, d in graph.degree()) / (num_nodes + 1) if num_nodes > 0 else 0
        }


class EnhancedSpatialGNN(nn.Module):
    """增强的空间关系图神经网络，处理自定义关系"""

    def __init__(self, node_feature_dim=8, edge_feature_dim=7,
                 hidden_dim=64, output_dim=128):
        super(EnhancedSpatialGNN, self).__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

        self.global_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x, edge_index, edge_attr, batch, global_features):
        x = self.node_encoder(x)

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)

        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        global_emb = self.global_encoder(global_features)

        combined = torch.cat([x, global_emb], dim=1)
        output = self.fusion_layer(combined)

        return output


class MapSimilarityCalculator:
    """地图相似度计算器"""

    def __init__(self, model=None):
        self.model = model or EnhancedSpatialGNN()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def compute_detailed_similarity(self, map1_data, map2_data):
        """计算详细的相似度，包括各个方面"""
        geom_sim = self._compute_geometric_similarity(map1_data, map2_data)
        topo_sim = self._compute_topological_similarity(map1_data, map2_data)
        rel_sim = self._compute_relationship_similarity(map1_data, map2_data)
        struct_sim = self._compute_structural_similarity(map1_data, map2_data)

        total_sim = (0.25 * geom_sim + 0.25 * topo_sim +
                     0.25 * rel_sim + 0.25 * struct_sim)

        return {
            'geometric_similarity': geom_sim,
            'topological_similarity': topo_sim,
            'relationship_similarity': rel_sim,
            'structural_similarity': struct_sim,
            'total_similarity': total_sim
        }

    def _compute_geometric_similarity(self, map1_data, map2_data):
        """计算几何相似度"""
        ratios = []

        for key in ['num_points', 'num_lines', 'num_polygons']:
            val1 = map1_data['global_features'][key]
            val2 = map2_data['global_features'][key]

            if val1 > 0 or val2 > 0:
                ratio = min(val1, val2) / (max(val1, val2) + 1e-6)
                ratios.append(ratio)

        length1 = map1_data['global_features']['total_line_length']
        length2 = map2_data['global_features']['total_line_length']

        area1 = map1_data['global_features']['total_polygon_area']
        area2 = map2_data['global_features']['total_polygon_area']

        if length1 > 0 or length2 > 0:
            length_ratio = min(length1, length2) / (max(length1, length2) + 1e-6)
            ratios.append(length_ratio)

        if area1 > 0 or area2 > 0:
            area_ratio = min(area1, area2) / (max(area1, area2) + 1e-6)
            ratios.append(area_ratio)

        return np.mean(ratios) if ratios else 0

    def _compute_topological_similarity(self, map1_data, map2_data):
        """计算拓扑相似度"""
        density1 = map1_data['global_features']['density']
        density2 = map2_data['global_features']['density']

        avg_degree1 = map1_data['global_features']['avg_degree']
        avg_degree2 = map2_data['global_features']['avg_degree']

        density_sim = 1 - abs(density1 - density2) / (max(density1, density2) + 1e-6)
        degree_sim = 1 - abs(avg_degree1 - avg_degree2) / (max(avg_degree1, avg_degree2) + 1e-6)

        return (density_sim + degree_sim) / 2

    def _compute_relationship_similarity(self, map1_data, map2_data):
        """计算关系相似度"""
        rel_types = ['gaze_relations', 'navigation_relations', 'connection_relations']
        similarities = []

        for rel_type in rel_types:
            count1 = map1_data['global_features'][rel_type]
            count2 = map2_data['global_features'][rel_type]

            if count1 > 0 or count2 > 0:
                sim = min(count1, count2) / (max(count1, count2) + 1e-6)
                similarities.append(sim)
            else:
                similarities.append(1.0)

        return np.mean(similarities)

    def _compute_structural_similarity(self, map1_data, map2_data):
        """使用简化的结构相似度计算"""
        # 简化版本：基于特征的余弦相似度
        feat1 = np.array([
            map1_data['global_features']['num_points'],
            map1_data['global_features']['num_lines'],
            map1_data['global_features']['num_polygons'],
            map1_data['global_features']['density'],
            map1_data['global_features']['avg_degree']
        ])

        feat2 = np.array([
            map2_data['global_features']['num_points'],
            map2_data['global_features']['num_lines'],
            map2_data['global_features']['num_polygons'],
            map2_data['global_features']['density'],
            map2_data['global_features']['avg_degree']
        ])

        # 归一化
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)

        if norm1 == 0 or norm2 == 0:
            return 0

        feat1_norm = feat1 / norm1
        feat2_norm = feat2 / norm2

        return np.dot(feat1_norm, feat2_norm)


class MapSimilaritySystem:
    """地图相似度计算系统主类"""

    def __init__(self, root_path: str):
        self.root_path = root_path
        self.loader = MapDataLoader(root_path)
        self.graph_builder = EnhancedSpatialRelationBuilder()
        self.feature_extractor = MapFeatureExtractor()
        self.model = EnhancedSpatialGNN()
        self.similarity_calculator = MapSimilarityCalculator(self.model)

    def prepare_map_data(self, group_name: str):
        """准备地图数据"""
        print(f"\n正在加载 {group_name} 的数据...")

        # 加载原始数据
        group_data = self.loader.load_group_data(group_name)

        # 转换为标准格式
        points, lines, polygons, relationships = self.loader.convert_to_standard_format(group_data)

        print(f"  - 点数量: {len(points)}")
        print(f"  - 线数量: {len(lines)}")
        print(f"  - 面数量: {len(polygons)}")
        print(f"  - 关系数量: {len(relationships)}")

        # 构建空间关系图
        graph = self.graph_builder.build_enhanced_spatial_graph(
            points, lines, polygons, relationships
        )

        print(f"  - 图节点数: {graph.number_of_nodes()}")
        print(f"  - 图边数: {graph.number_of_edges()}")

        # 提取特征
        node_features = self.feature_extractor.extract_node_features(graph)
        edge_features = self.feature_extractor.extract_edge_features(graph)
        global_features = self.feature_extractor.extract_global_features(
            graph, points, lines, polygons
        )

        return {
            'name': group_name,
            'points': points,
            'lines': lines,
            'polygons': polygons,
            'relationships': relationships,
            'graph': graph,
            'node_features': node_features,
            'edge_features': edge_features,
            'global_features': global_features,
            'group_data': group_data
        }

    def compute_map_similarity(self, map1_data, map2_data):
        """计算两个地图的相似度"""
        print(f"\n计算 {map1_data['name']} 与 {map2_data['name']} 的相似度...")

        # 计算详细相似度
        similarity_results = self.similarity_calculator.compute_detailed_similarity(
            map1_data, map2_data
        )

        return similarity_results

    def visualize_comparison(self, map1_data, map2_data, similarity_results,
                             output_path=None):
        """可视化地图比较结果"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # 绘制第一个地图
        self._plot_map(axes[0], map1_data, f"{map1_data['name']}")

        # 绘制第二个地图
        self._plot_map(axes[1], map2_data, f"{map2_data['name']}")

        # 添加相似度信息
        sim_text = (
            f"几何相似度: {similarity_results['geometric_similarity']:.3f}\n"
            f"拓扑相似度: {similarity_results['topological_similarity']:.3f}\n"
            f"关系相似度: {similarity_results['relationship_similarity']:.3f}\n"
            f"结构相似度: {similarity_results['structural_similarity']:.3f}\n"
            f"总体相似度: {similarity_results['total_similarity']:.3f}"
        )

        fig.text(0.5, 0.02, sim_text, ha='center', fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

        plt.tight_layout(rect=[0, 0.1, 1, 0.98])

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[√] 已保存比较图: {output_path}")
        else:
            plt.show()

        plt.close()

    def _plot_map(self, ax, map_data, title):
        """绘制单个地图 - 使用您的风格"""
        # 绘制多边形
        for poly in map_data['polygons']:
            if len(poly) > 2:
                xs, ys = zip(*poly)
                ax.fill(xs, ys, facecolor="#ffe4b5", edgecolor="#555555", alpha=0.6)

        # 绘制线
        for line in map_data['lines']:
            if len(line) > 1:
                xs, ys = zip(*line)
                ax.plot(xs, ys, color="#1f78b4", linewidth=1.5)

        # 绘制点
        for pt in map_data['points']:
            ax.scatter(pt[0], pt[1], color="#e31a1c", s=40)

        # 绘制关系箭头
        color_map = {
            'Gaze': 'purple',
            'Navigation': 'green',
            'Connection': 'orange'
        }
        # 创建名称到几何对象的映射
        geom_mapping = {}

        for i, point in enumerate(map_data['points']):
            geom_mapping[('point', i)] = point

        for i, line in enumerate(map_data['lines']):
            if len(line) > 1:
                centroid = np.mean(line, axis=0)
                geom_mapping[('line', i)] = tuple(centroid)

        for i, poly in enumerate(map_data['polygons']):
            if len(poly) > 2:
                centroid = np.mean(poly, axis=0)
                geom_mapping[('polygon', i)] = tuple(centroid)

        # 绘制关系
        for rel in map_data['relationships']:
            src_key = rel['source']
            dst_key = rel['target']
            rel_type = rel['type']

            if src_key in geom_mapping and dst_key in geom_mapping:
                src_pos = geom_mapping[src_key]
                dst_pos = geom_mapping[dst_key]

                # 绘制箭头
                arrow = patches.FancyArrowPatch(
                    src_pos, dst_pos,
                    arrowstyle='->',
                    color=color_map.get(rel_type, 'black'),
                    linewidth=1.5,
                    mutation_scale=15,
                    alpha=0.7
                )
                ax.add_patch(arrow)

        # 图例
        legend_elems = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor="#e31a1c",
                   markersize=8, label='Point'),
            Line2D([0], [0], color="#1f78b4", lw=2, label='Polyline'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor="#ffe4b5",
                   markersize=10, label='Polygon'),
        ]

        # 如果有关系，添加关系图例
        if map_data['relationships']:
            legend_elems.extend([
                Line2D([0], [0], color='purple', lw=2, label='Gaze'),
                Line2D([0], [0], color='green', lw=2, label='Navigation'),
                Line2D([0], [0], color='orange', lw=2, label='Connection')
            ])

        ax.legend(handles=legend_elems, loc='upper right')
        ax.set_aspect("equal", "box")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{title} (merged)")
        ax.grid(True, alpha=0.3)

    def generate_similarity_report(self, results, output_path):
        """生成相似度报告"""
        report = {
            'timestamp': str(np.datetime64('now')),
            'results': results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"[√] 已保存相似度报告: {output_path}")

    def run_complete_analysis(self):
        """运行完整的相似度分析"""
        print("=" * 60)
        print("地图相似度分析系统")
        print("=" * 60)

        # 准备数据
        map_data = {}

        for group in ["Group1", "Group2", "RealMap"]:
            gdir = os.path.join(self.root_path, group)
            if os.path.isdir(gdir):
                try:
                    map_data[group] = self.prepare_map_data(group)
                except Exception as e:
                    print(f"加载 {group} 失败: {e}")
                    continue

        if len(map_data) < 2 or 'RealMap' not in map_data:
            print("数据不足，无法进行相似度分析")
            return

        # 计算相似度
        results = {}

        # Group1 vs RealMap
        if 'Group1' in map_data:
            print("\n" + "=" * 40)
            sim_g1_real = self.compute_map_similarity(map_data['Group1'], map_data['RealMap'])
            results['Group1_vs_RealMap'] = sim_g1_real

        # Group2 vs RealMap
        if 'Group2' in map_data:
            print("\n" + "=" * 40)
            sim_g2_real = self.compute_map_similarity(map_data['Group2'], map_data['RealMap'])
            results['Group2_vs_RealMap'] = sim_g2_real

        # Group1 vs Group2
        if 'Group1' in map_data and 'Group2' in map_data:
            print("\n" + "=" * 40)
            sim_g1_g2 = self.compute_map_similarity(map_data['Group1'], map_data['Group2'])
            results['Group1_vs_Group2'] = sim_g1_g2

        # 打印结果摘要
        print("\n" + "=" * 60)
        print("相似度分析结果摘要")
        print("=" * 60)

        for comparison, sim_results in results.items():
            print(f"\n{comparison}:")
            print(f"  - 几何相似度: {sim_results['geometric_similarity']:.3f}")
            print(f"  - 拓扑相似度: {sim_results['topological_similarity']:.3f}")
            print(f"  - 关系相似度: {sim_results['relationship_similarity']:.3f}")
            print(f"  - 结构相似度: {sim_results['structural_similarity']:.3f}")
            print(f"  - 总体相似度: {sim_results['total_similarity']:.3f}")

        # 生成可视化
        print("\n生成可视化结果...")

        # 绘制单独的地图
        for group, data in map_data.items():
            self.plot_single_map(data, os.path.join(OUT_DIR, f"{group}.png"))

        # 绘制比较图
        if 'Group1' in map_data and 'RealMap' in map_data:
            self.visualize_comparison(
                map_data['Group1'], map_data['RealMap'], results['Group1_vs_RealMap'],
                os.path.join(OUT_DIR, "comparison_Group1_RealMap.png")
            )

        if 'Group2' in map_data and 'RealMap' in map_data:
            self.visualize_comparison(
                map_data['Group2'], map_data['RealMap'], results['Group2_vs_RealMap'],
                os.path.join(OUT_DIR, "comparison_Group2_RealMap.png")
            )

        if 'Group1' in map_data and 'Group2' in map_data:
            self.visualize_comparison(
                map_data['Group1'], map_data['Group2'], results['Group1_vs_Group2'],
                os.path.join(OUT_DIR, "comparison_Group1_Group2.png")
            )

        # 生成报告
        self.generate_similarity_report(
            results,
            os.path.join(OUT_DIR, "similarity_report.json")
        )

        # 生成条形图比较
        self._plot_similarity_bars(results, os.path.join(OUT_DIR, "similarity_comparison.png"))

        # 确定哪个组与RealMap更相似
        print("\n" + "=" * 60)
        print("结论")
        print("=" * 60)

        g1_sim = results.get('Group1_vs_RealMap', {}).get('total_similarity', 0)
        g2_sim = results.get('Group2_vs_RealMap', {}).get('total_similarity', 0)

        if g1_sim > g2_sim:
            print(f"Group1 与 RealMap 更相似 (相似度: {g1_sim:.3f} vs {g2_sim:.3f})")
        elif g2_sim > g1_sim:
            print(f"Group2 与 RealMap 更相似 (相似度: {g2_sim:.3f} vs {g1_sim:.3f})")
        else:
            print(f"Group1 和 Group2 与 RealMap 的相似度相同 (相似度: {g1_sim:.3f})")

        print(f"\n分析完成！所有结果已保存到: {OUT_DIR}")

    def plot_single_map(self, map_data, output_path):
        """绘制单个地图 - 使用您的风格"""
        fig, ax = plt.subplots(figsize=(10, 10))
        self._plot_map(ax, map_data, map_data['name'])
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[√] 已保存 {output_path}")

    def _plot_similarity_bars(self, results, output_path):
        """绘制相似度条形图比较"""
        fig, ax = plt.subplots(figsize=(12, 8))

        comparisons = list(results.keys())
        metrics = ['geometric_similarity', 'topological_similarity',
                   'relationship_similarity', 'structural_similarity', 'total_similarity']

        x = np.arange(len(comparisons))
        width = 0.15

        colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3']

        for i, metric in enumerate(metrics):
            values = [results[comp][metric] for comp in comparisons]
            offset = (i - 2) * width
            bars = ax.bar(x + offset, values, width,
                          label=metric.replace('_', ' ').title(),
                          color=colors[i])

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=10)

        ax.set_xlabel('比较对象', fontsize=12)
        ax.set_ylabel('相似度得分', fontsize=12)
        ax.set_title('地图相似度比较结果', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparisons)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[√] 已保存相似度条形图: {output_path}")


# 简化版本的处理函数，用于直接生成原始风格的合并图
def process_group_simple(group_path, group_name):
    """把一个组(Group1/Group2/RealMap)内所有 Sample 的点线面合并到一张图"""
    areas, lines, points = [], [], []

    for sample in os.listdir(group_path):
        spath = os.path.join(group_path, sample)
        if not os.path.isdir(spath):
            continue

        area_shp = first_shp(os.path.join(spath, "my_zip_area"))
        line_shp = first_shp(os.path.join(spath, "my_zip_line"))
        point_shp = first_shp(os.path.join(spath, "my_zip_point"))

        if area_shp:  areas += build_geom_list(area_shp)
        if line_shp:  lines += build_geom_list(line_shp)
        if point_shp: points += build_geom_list(point_shp)

    if not (areas or lines or points):
        print(f"[跳过] {group_name} 没找到任何几何要素")
        return

    # ---------- 绘图 ----------
    fig, ax = plt.subplots(figsize=(10, 10))

    for poly in areas:
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, facecolor="#ffe4b5", edgecolor="#555555", alpha=0.6)

    for ln in lines:
        xs, ys = ln.xy
        ax.plot(xs, ys, color="#1f78b4", linewidth=1.5)

    for pt in points:
        x, y = pt.coords[0]
        ax.scatter(x, y, color="#e31a1c", s=40)

    # 图例
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#e31a1c",
               markersize=8, label='Point'),
        Line2D([0], [0], color="#1f78b4", lw=2, label='Polyline'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor="#ffe4b5",
               markersize=10, label='Polygon'),
    ]
    ax.legend(handles=legend_elems, loc='upper right')
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{group_name} (merged)")
    plt.tight_layout()
    out_file = os.path.join(OUT_DIR, f"{group_name}_simple.png".replace(" ", "_"))
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[√] 已保存 {out_file}")


# ------------------------------------------------------------
# 主程序
def main():
    """主函数"""
    # 创建系统实例
    system = MapSimilaritySystem(ROOT)

    # 运行完整分析
    system.run_complete_analysis()

    # 额外生成简单版本的合并图（保持原始风格）
    print("\n生成简单版本的合并图...")
    for group in ["Group1", "Group2", "RealMap"]:
        gdir = os.path.join(ROOT, group)
        if os.path.isdir(gdir):
            process_group_simple(gdir, group)

    print("\n全部完成!合并图已输出到", OUT_DIR)


if __name__ == "__main__":
    main()