import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec
from shapely.geometry import Point, LineString, Polygon as ShapelyPolygon
from shapely.ops import nearest_points
import rtree
import networkx as nx
from sklearn.preprocessing import StandardScaler
import math
from scipy.spatial.distance import directed_hausdorff, cdist

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# ==============================
# 1. 辅助函数：空间关系计算
# ==============================

def distance(p1, p2):
    """计算两点之间的欧氏距离"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def normalize_vector(v):
    """将向量标准化为单位向量"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def compute_bbox(geometry):
    """计算几何对象的边界框 [minx, miny, maxx, maxy]"""
    if isinstance(geometry, tuple):  # 点
        return [geometry[0], geometry[1], geometry[0], geometry[1]]
    elif isinstance(geometry, list) and all(isinstance(p, tuple) for p in geometry):  # 线或多边形
        x_coords = [p[0] for p in geometry]
        y_coords = [p[1] for p in geometry]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    else:
        raise ValueError("不支持的几何类型")


def compute_centroid(points):
    """计算点集的质心"""
    x_sum = sum(p[0] for p in points)
    y_sum = sum(p[1] for p in points)
    return (x_sum / len(points), y_sum / len(points))


def compute_angle(p1, p2):
    """计算从p1到p2的方向角（弧度）"""
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def compute_direction_coverage(direction_vectors, num_bins=8):
    """计算方向向量的覆盖情况，返回0-1的分数"""
    # 将方向分成num_bins个区间
    bins = [0] * num_bins
    bin_size = 2 * math.pi / num_bins

    for v in direction_vectors:
        angle = math.atan2(v[1], v[0])
        if angle < 0:
            angle += 2 * math.pi
        bin_idx = min(int(angle / bin_size), num_bins - 1)
        bins[bin_idx] += 1

    # 计算方向覆盖度
    non_empty_bins = sum(1 for b in bins if b > 0)
    return non_empty_bins / num_bins


def compute_frechet_distance(line1, line2):
    """计算两条线之间的弗雷歇距离"""
    # 使用简化版本，基于hausdorff距离
    return max(directed_hausdorff(line1, line2)[0], directed_hausdorff(line2, line1)[0])


def compute_hausdorff_distance(poly1, poly2):
    """计算两个多边形之间的豪斯多夫距离"""
    return max(directed_hausdorff(poly1, poly2)[0], directed_hausdorff(poly2, poly1)[0])


def compute_iou(poly1, poly2):
    """计算两个多边形的交并比"""
    # 转换为Shapely多边形
    shapely_poly1 = ShapelyPolygon(poly1)
    shapely_poly2 = ShapelyPolygon(poly2)

    # 计算交集和并集面积
    intersection_area = shapely_poly1.intersection(shapely_poly2).area
    union_area = shapely_poly1.union(shapely_poly2).area

    # 防止除零
    if union_area == 0:
        return 0

    return intersection_area / union_area


def compute_point_line_distance(point, line):
    """计算点到线的最短距离"""
    shapely_point = Point(point)
    shapely_line = LineString(line)
    return shapely_point.distance(shapely_line)


def compute_point_polygon_relation(point, polygon):
    """计算点与多边形的关系"""
    shapely_point = Point(point)
    shapely_polygon = ShapelyPolygon(polygon)

    # 检查点是否在多边形内部或边界上
    is_inside = shapely_polygon.contains(shapely_point)
    is_on_boundary = shapely_polygon.boundary.distance(shapely_point) < 1e-8

    # 计算点到多边形的距离
    if is_inside:
        distance = 0
        relation_type = 'inside'
    elif is_on_boundary:
        distance = 0
        relation_type = 'on_boundary'
    else:
        distance = shapely_point.distance(shapely_polygon)
        relation_type = 'outside'

    return {'type': relation_type, 'distance': distance}


def compute_line_line_relation(line1, line2, threshold=1e-8):
    """计算两条线之间的关系"""
    shapely_line1 = LineString(line1)
    shapely_line2 = LineString(line2)

    # 检查线是否相交
    intersection = shapely_line1.intersection(shapely_line2)
    has_intersection = not intersection.is_empty

    # 确定关系类型
    if has_intersection:
        if intersection.geom_type == 'Point':
            relation_type = 'touch'
            intersection_points = [(intersection.x, intersection.y)]
        else:
            relation_type = 'overlap'
            # 提取所有交点（如果是LineString类型的交集）
            if intersection.geom_type == 'LineString':
                intersection_points = list(intersection.coords)
            else:
                # 对于MultiPoint等其他类型
                intersection_points = []
                for geom in intersection.geoms:
                    if geom.geom_type == 'Point':
                        intersection_points.append((geom.x, geom.y))
    else:
        # 检查是否平行
        if shapely_line1.distance(shapely_line2) < threshold:
            relation_type = 'parallel'
        else:
            relation_type = 'disjoint'
        intersection_points = []

    return {
        'type': relation_type,
        'interaction': has_intersection,
        'intersections': intersection_points
    }


def compute_line_polygon_relation(line, polygon):
    """计算线与多边形的关系"""
    try:
        # 尝试创建有效的几何对象
        shapely_line = LineString(line)
        shapely_polygon = ShapelyPolygon(polygon)

        # 确保几何对象是有效的
        if not shapely_line.is_valid:
            shapely_line = shapely_line.buffer(0)  # 尝试修复无效线
        if not shapely_polygon.is_valid:
            shapely_polygon = shapely_polygon.buffer(0)  # 尝试修复无效多边形

        # 计算距离（这是安全的操作）
        distance = shapely_line.distance(shapely_polygon.boundary)

        try:
            # 检查相交关系（可能出现拓扑异常）
            intersection = shapely_line.intersection(shapely_polygon)
            has_intersection = not intersection.is_empty

            # 计算线在多边形内部的部分长度
            if has_intersection:
                if intersection.geom_type == 'LineString':
                    inside_length = intersection.length
                    relation_type = 'crosses'
                elif intersection.geom_type == 'MultiLineString':
                    inside_length = sum(geom.length for geom in intersection.geoms)
                    relation_type = 'crosses'
                elif intersection.geom_type == 'Point':
                    inside_length = 0
                    relation_type = 'touches'
                elif intersection.geom_type == 'MultiPoint':
                    inside_length = 0
                    relation_type = 'touches'
                else:
                    inside_length = 0
                    relation_type = 'unknown'

                # 计算线与多边形的外接
                # 如果线完全在多边形内部
                if shapely_polygon.contains(shapely_line):
                    relation_type = 'within'
                # 如果线在多边形边界上
                elif shapely_polygon.boundary.intersection(shapely_line).length > 0:
                    relation_type = 'on_boundary'
            else:
                inside_length = 0
                relation_type = 'disjoint'

        except Exception as e:
            # 如果计算相交关系出现异常，使用安全的近似方法
            print(f"Warning: Intersection calculation failed: {e}")
            # 使用默认值
            inside_length = 0
            relation_type = 'unknown'
            has_intersection = False

        # 计算inside_ratio
        total_length = shapely_line.length
        inside_ratio = inside_length / total_length if total_length > 0 else 0

    except Exception as e:
        # 如果出现任何异常，使用默认值
        print(f"Warning: Line-polygon relation calculation failed: {e}")
        distance = float('inf')
        inside_length = 0
        relation_type = 'error'
        total_length = 0
        inside_ratio = 0

    return {
        'type': relation_type,
        'distance': distance,
        'inside_length': inside_length,
        'total_length': total_length if 'total_length' in locals() else 0,
        'inside_ratio': inside_ratio
    }


def compute_polygon_polygon_relation(poly1, poly2):
    """计算两个多边形之间的关系"""
    try:
        # 尝试创建有效的几何对象
        shapely_poly1 = ShapelyPolygon(poly1)
        shapely_poly2 = ShapelyPolygon(poly2)

        # 确保几何对象是有效的
        if not shapely_poly1.is_valid:
            shapely_poly1 = shapely_poly1.buffer(0)
        if not shapely_poly2.is_valid:
            shapely_poly2 = shapely_poly2.buffer(0)

        # 计算距离（这是安全的操作）
        distance = shapely_poly1.distance(shapely_poly2)

        try:
            # 检查相交关系（可能出现拓扑异常）
            intersection = shapely_poly1.intersection(shapely_poly2)
            has_intersection = not intersection.is_empty

            # 计算交集面积
            if has_intersection:
                intersection_area = intersection.area
                relation_type = 'overlap'

                # 检查是否一个完全包含另一个
                if shapely_poly1.contains(shapely_poly2):
                    relation_type = 'contains'
                elif shapely_poly2.contains(shapely_poly1):
                    relation_type = 'within'
                # 检查是否仅在边界相交
                elif intersection.area == 0:
                    relation_type = 'touch'
            else:
                intersection_area = 0
                relation_type = 'disjoint'

            # 计算交并比
            union_area = shapely_poly1.union(shapely_poly2).area
            iou = intersection_area / union_area if union_area > 0 else 0

        except Exception as e:
            # 如果计算相交关系出现异常，使用安全的近似方法
            print(f"Warning: Intersection calculation failed: {e}")
            # 使用默认值
            intersection_area = 0
            relation_type = 'unknown'
            union_area = shapely_poly1.area + shapely_poly2.area
            iou = 0

    except Exception as e:
        # 如果出现任何异常，使用默认值
        print(f"Warning: Polygon-polygon relation calculation failed: {e}")
        distance = float('inf')
        intersection_area = 0
        relation_type = 'error'
        union_area = 0
        iou = 0

    return {
        'type': relation_type,
        'distance': distance,
        'intersection_area': intersection_area,
        'union_area': union_area,
        'iou': iou
    }


# ==============================
# 2. 空间索引和邻近搜索
# ==============================

class SpatialIndex:
    """空间索引类，用于快速查询空间对象"""

    def __init__(self, dimension=2):
        self.index = rtree.index.Index()
        self.objects = {}
        self.dimension = dimension
        self.count = 0

    def insert(self, obj, obj_type):
        """插入对象到空间索引"""
        bbox = compute_bbox(obj)
        self.index.insert(self.count, bbox)
        self.objects[self.count] = {
            'type': obj_type,
            'geometry': obj
        }
        self.count += 1

def query_nearest(self, point, n=1):
    """查询距离指定点最近的n个对象"""
    bbox = [point[0], point[1], point[0], point[1]]  # 点的边界框
    nearest_ids = list(self.index.nearest(bbox, n))
    return [self.objects[i] for i in nearest_ids]

def query_radius(self, point, radius):
    """查询指定半径内的所有对象"""
    bbox = [
        point[0] - radius,
        point[1] - radius,
        point[0] + radius,
        point[1] + radius
    ]
    ids = list(self.index.intersection(bbox))

    # 进一步筛选，确保真正在半径内
    result = []
    for i in ids:
        obj = self.objects[i]

        if obj['type'] == 'point':
            if distance(point, obj['geometry']) <= radius:
                result.append(obj)

        elif obj['type'] == 'line':
            if compute_point_line_distance(point, obj['geometry']) <= radius:
                result.append(obj)

        elif obj['type'] == 'polygon':
            relation = compute_point_polygon_relation(point, obj['geometry'])
            if relation['distance'] <= radius:
                result.append(obj)

    return result

def get_all_by_type(self, obj_type):
    """获取所有指定类型的对象"""
    return [obj for obj_id, obj in self.objects.items() if obj['type'] == obj_type]

    # ==============================
    # 3. 空间关系图构建
    # ==============================

def build_spatial_relation_graph(points, lines, polygons, distance_threshold=2.0):
    """
    构建空间关系图，捕获不同几何对象间的空间关系

    参数:
        points: 点列表，每个元素是(x, y)元组
        lines: 线列表，每个元素是点的列表
        polygons: 多边形列表，每个元素是点的列表
        distance_threshold: 空间关系的距离阈值

    返回:
        NetworkX图对象和关系统计信息
    """
    graph = nx.Graph()

    # 添加所有对象作为节点
    for i, p in enumerate(points):
        graph.add_node(f'p{i}', type='point', geometry=p, coords=p)

        for i, l in enumerate(lines):
            # 计算线的基本特征
            # 修改这里，直接计算距离而不是调用distance函数
            length = sum(
                np.sqrt((l[j][0] - l[j + 1][0]) ** 2 + (l[j][1] - l[j + 1][1]) ** 2) for j in range(len(l) - 1))
            centroid = compute_centroid(l)

            graph.add_node(f'l{i}', type='line', geometry=l,
                           coords=centroid, length=length)

    #可能会有报错，余下代码不变
    for i, poly in enumerate(polygons):
        # 计算多边形的基本特征
        shapely_poly = ShapelyPolygon(poly)
        area = shapely_poly.area
        perimeter = shapely_poly.length
        centroid = compute_centroid(poly)

        graph.add_node(f'poly{i}', type='polygon', geometry=poly,
                       coords=centroid, area=area, perimeter=perimeter)

    # 统计各种关系的数量
    stats = {
        'point_line': 0,
        'point_polygon': 0,
        'line_line': 0,
        'line_polygon': 0,
        'polygon_polygon': 0
    }
    # 添加点与线的关系
    for i, p in enumerate(points):
        for j, l in enumerate(lines):
            distance = compute_point_line_distance(p, l)
            if distance <= distance_threshold:
                graph.add_edge(f'p{i}', f'l{j}',
                               relation_type='point_line',
                               distance=distance)
                stats['point_line'] += 1

    # 添加点与多边形的关系
    for i, p in enumerate(points):
        for j, poly in enumerate(polygons):
            relation = compute_point_polygon_relation(p, poly)
            if relation['distance'] <= distance_threshold or relation['type'] in ['inside', 'on_boundary']:
                graph.add_edge(f'p{i}', f'poly{j}',
                               relation_type='point_polygon',
                               distance=relation['distance'],
                               spatial_relation=relation['type'])
                stats['point_polygon'] += 1

    # 添加线与线的关系
    for i, l1 in enumerate(lines):
        for j, l2 in enumerate(lines):
            if i < j:  # 避免重复
                relation = compute_line_line_relation(l1, l2)
                if relation['interaction'] or relation['type'] == 'parallel':
                    graph.add_edge(f'l{i}', f'l{j}',
                                   relation_type='line_line',
                                   spatial_relation=relation['type'],
                                   intersections=relation['intersections'])
                    stats['line_line'] += 1

    # 添加线与多边形的关系
    for i, l in enumerate(lines):
        for j, poly in enumerate(polygons):
            relation = compute_line_polygon_relation(l, poly)
            if relation['distance'] <= distance_threshold or relation['type'] in ['crosses', 'within',
                                                                                  'on_boundary']:
                graph.add_edge(f'l{i}', f'poly{j}',
                               relation_type='line_polygon',
                               distance=relation['distance'],
                               spatial_relation=relation['type'],
                               inside_ratio=relation['inside_ratio'])
                stats['line_polygon'] += 1

    # 添加多边形与多边形的关系
    for i, poly1 in enumerate(polygons):
        for j, poly2 in enumerate(polygons):
            if i < j:  # 避免重复
                relation = compute_polygon_polygon_relation(poly1, poly2)
                if relation['distance'] <= distance_threshold or relation['type'] in ['overlap', 'contains',
                                                                                      'within', 'touch']:
                    graph.add_edge(f'poly{i}', f'poly{j}',
                                   relation_type='polygon_polygon',
                                   distance=relation['distance'],
                                   spatial_relation=relation['type'],
                                   iou=relation['iou'])
                    stats['polygon_polygon'] += 1

    return graph, stats

    # ==============================
    # 4. 图数据转换为PyTorch Geometric格式
    # ==============================

def convert_nx_to_pyg(nx_graph):
    """将NetworkX图转换为PyTorch Geometric数据对象"""
    # 节点特征映射
    node_mapping = {}
    node_features = []
    node_types = []
    edge_index = []
    edge_features = []

    # 确定最大特征长度
    max_feature_length = 0

    # 预处理节点以确定最大特征长度
    for node, data in nx_graph.nodes(data=True):
        if data['type'] == 'point':
            feature_length = 2  # x, y坐标
        elif data['type'] == 'line':
            feature_length = 3  # x, y坐标 + 长度
        elif data['type'] == 'polygon':
            feature_length = 4  # x, y坐标 + 面积 + 周长
        else:
            feature_length = 2  # 默认长度

        max_feature_length = max(max_feature_length, feature_length)

    # 如果图为空，设置默认最大特征长度
    if max_feature_length == 0:
        max_feature_length = 4  # 默认使用最大可能长度

    # 处理节点
    for i, (node, data) in enumerate(nx_graph.nodes(data=True)):
        node_mapping[node] = i

        # 根据节点类型构建特征，并确保所有特征向量长度一致
        if data['type'] == 'point':
            # 点特征：x, y坐标
            features = [data['coords'][0], data['coords'][1]]
            # 填充到最大长度
            features.extend([0.0] * (max_feature_length - len(features)))
            node_types.append(0)  # 0表示点

        elif data['type'] == 'line':
            # 线特征：质心坐标，长度
            features = [data['coords'][0], data['coords'][1], data.get('length', 0.0)]
            # 填充到最大长度
            features.extend([0.0] * (max_feature_length - len(features)))
            node_types.append(1)  # 1表示线

        elif data['type'] == 'polygon':
            # 多边形特征：质心坐标，面积，周长
            features = [data['coords'][0], data['coords'][1],
                        data.get('area', 0.0), data.get('perimeter', 0.0)]
            # 填充到最大长度（如果需要）
            features.extend([0.0] * (max_feature_length - len(features)))
            node_types.append(2)  # 2表示多边形

        node_features.append(features[:max_feature_length])  # 确保不超过最大长度

    # 处理边
    for u, v, data in nx_graph.edges(data=True):
        # 检查节点是否在映射中
        if u not in node_mapping or v not in node_mapping:
            print(f"Warning: Edge ({u}, {v}) references missing nodes, skipping")
            continue

        edge_index.append([node_mapping[u], node_mapping[v]])
        edge_index.append([node_mapping[v], node_mapping[u]])  # 双向边

        # 边特征
        if data['relation_type'] == 'point_line':
            feat = [data.get('distance', 0.0), 0, 0, 0, 0]

        elif data['relation_type'] == 'point_polygon':
            # 将空间关系编码为数值
            relation_code = {
                'inside': 1,
                'on_boundary': 2,
                'outside': 3
            }.get(data.get('spatial_relation', 'outside'), 0)

            feat = [data.get('distance', 0.0), 0, 0, relation_code, 0]

        elif data['relation_type'] == 'line_line':
            # 将空间关系编码为数值
            relation_code = {
                'touch': 1,
                'overlap': 2,
                'parallel': 3,
                'disjoint': 4
            }.get(data.get('spatial_relation', 'disjoint'), 0)

            # 计算交点数量
            intersections_count = len(data.get('intersections', []))

            feat = [0, relation_code, intersections_count, 0, 0]

        elif data['relation_type'] == 'line_polygon':
            # 将空间关系编码为数值
            relation_code = {
                'crosses': 1,
                'within': 2,
                'on_boundary': 3,
                'touches': 4,
                'disjoint': 5
            }.get(data.get('spatial_relation', 'disjoint'), 0)

            feat = [data.get('distance', 0.0), 0, 0, 0, data.get('inside_ratio', 0.0)]

        elif data['relation_type'] == 'polygon_polygon':
            # 将空间关系编码为数值
            relation_code = {
                'overlap': 1,
                'contains': 2,
                'within': 3,
                'touch': 4,
                'disjoint': 5
            }.get(data.get('spatial_relation', 'disjoint'), 0)

            feat = [data.get('distance', 0.0), 0, 0, 0, data.get('iou', 0.0)]

        else:
            feat = [0, 0, 0, 0, 0]

        edge_features.append(feat)
        edge_features.append(feat)  # 双向边使用相同特征

    # 确保有有效的节点和边
    if not node_features:
        print("Warning: Graph has no valid nodes, creating a minimal valid graph")
        # 创建一个具有单一节点的图
        node_features = [[0.0] * max_feature_length]
        node_types = [0]  # 点类型

    if not edge_index:
        print("Warning: Graph has no valid edges, creating a self-loop")
        # 创建一个自环边
        edge_index = [[0, 0]]
        edge_features = [[0, 0, 0, 0, 0]]

    # 转换为tensor
    node_features = torch.tensor(node_features, dtype=torch.float)
    node_types = torch.tensor(node_types, dtype=torch.long)

    # 确保边索引列表不为空
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(edge_features, dtype=torch.float)
    else:
        # 创建空的边索引张量
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_features = torch.zeros((0, 5), dtype=torch.float)

    # 创建PyG数据对象
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features,
        node_type=node_types
    )

    return data

        # ==============================
        # 5. 形状相似度计算模型
        # ==============================

class NodeTypeEncoder(nn.Module):
    """节点类型编码器，为不同类型的节点使用不同的编码网络"""

    def __init__(self, point_dim, line_dim, polygon_dim, hidden_dim):
        super(NodeTypeEncoder, self).__init__()

        # 点编码器
        self.point_encoder = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 线编码器
        self.line_encoder = nn.Sequential(
            nn.Linear(line_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 多边形编码器
        self.polygon_encoder = nn.Sequential(
            nn.Linear(polygon_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        def forward(self, x, node_type):
            """
            编码不同类型的节点

            参数:
                x: 节点特征张量
                node_type: 节点类型张量 (0=点, 1=线, 2=多边形)

            返回:
                编码后的节点特征
            """
            encoded_features = torch.zeros((x.size(0), self.point_encoder[0].out_features), device=x.device)

            # 为点节点应用点编码器
            point_mask = (node_type == 0)
            if point_mask.any():
                point_features = x[point_mask]
                encoded_features[point_mask] = self.point_encoder(point_features)

            # 为线节点应用线编码器
            line_mask = (node_type == 1)
            if line_mask.any():
                line_features = x[line_mask]
                encoded_features[line_mask] = self.line_encoder(line_features)

            # 为多边形节点应用多边形编码器
            polygon_mask = (node_type == 2)
            if polygon_mask.any():
                polygon_features = x[polygon_mask]
                encoded_features[polygon_mask] = self.polygon_encoder(polygon_features)

            return encoded_features

class SpatialRelationGNN(nn.Module):
    """空间关系图神经网络，基于不同节点类型和边关系处理图数据"""

    def __init__(self, point_dim=2, line_dim=3, polygon_dim=4,
                 hidden_dim=64, edge_dim=5, output_dim=128):
        super(SpatialRelationGNN, self).__init__()

        # 节点类型编码器
        self.node_type_encoder = NodeTypeEncoder(
            point_dim, line_dim, polygon_dim, hidden_dim
        )

        # 边特征处理
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 图卷积层
        self.conv1 = GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
        self.conv3 = GATConv(hidden_dim, output_dim, edge_dim=hidden_dim)

        # 池化后的MLP
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        """
        前向传播

        参数:
            data: PyG数据对象

        返回:
            图的嵌入表示
        """
        x, edge_index, edge_attr, node_type, batch = data.x, data.edge_index, data.edge_attr, data.node_type, data.batch

        # 如果batch未提供，创建默认batch (所有节点属于同一个图)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 根据节点类型编码节点特征
        x = self.node_type_encoder(x, node_type)

        # 编码边特征
        edge_attr = self.edge_encoder(edge_attr)

        # 图卷积层
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv3(x, edge_index, edge_attr)

        # 全局池化
        x = global_mean_pool(x, batch)

        # 应用MLP
        x = self.mlp(x)

        return x


class ShapeSimilarityModel(nn.Module):
    """形状相似度计算模型，比较两个场景的相似度"""

    def __init__(self, output_dim=128, hidden_dim=64):
        super(ShapeSimilarityModel, self).__init__()

        # 空间关系图神经网络
        self.gnn = SpatialRelationGNN(output_dim=output_dim)

        # 点集相似度网络
        self.point_similarity = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 线集相似度网络
        self.line_similarity = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 多边形集相似度网络
        self.polygon_similarity = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 综合相似度网络
        self.overall_similarity = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def compute_embeddings(self, data):
        """计算场景的嵌入表示"""
        return self.gnn(data)

    def compute_similarity_by_type(self, embedding1, embedding2, node_type1, node_type2):
        """
        根据节点类型计算相似度

        参数:
            embedding1: 第一个场景的嵌入
            embedding2: 第二个场景的嵌入
            node_type1: 第一个场景的节点类型
            node_type2: 第二个场景的节点类型

        返回:
            点、线、面的相似度
        """
        # 拼接两个场景的嵌入
        combined = torch.cat([embedding1, embedding2], dim=1)

        # 计算点相似度
        point_sim = self.point_similarity(combined)

        # 计算线相似度
        line_sim = self.line_similarity(combined)

        # 计算多边形相似度
        polygon_sim = self.polygon_similarity(combined)

        return point_sim, line_sim, polygon_sim

    def forward(self, data1, data2):
        """
        计算两个场景的相似度

        参数:
            data1: 第一个场景的PyG数据对象
            data2: 第二个场景的PyG数据对象

        返回:
            包含点、线、面和整体相似度的字典
        """
        # 计算场景嵌入
        embedding1 = self.compute_embeddings(data1)
        embedding2 = self.compute_embeddings(data2)

        # 计算各类型相似度
        point_sim, line_sim, polygon_sim = self.compute_similarity_by_type(
            embedding1, embedding2, data1.node_type, data2.node_type
        )

        # 计算整体相似度
        overall_sim = self.overall_similarity(
            torch.cat([point_sim, line_sim, polygon_sim], dim=1)
        )

        return {
            'point_similarity': point_sim,
            'line_similarity': line_sim,
            'polygon_similarity': polygon_sim,
            'overall_similarity': overall_sim
        }


# ==============================
# 6. 相似度计算函数
# ==============================

def compute_point_set_similarity(reference_point, point_set, max_distance=10.0):
    """
    计算一个点与点集的相似度

    参数:
        reference_point: 参考点，(x, y)坐标
        point_set: 点集，列表包含多个(x, y)坐标
        max_distance: 最大距离阈值，用于归一化

    返回:
        相似度分数 (0-1)
    """
    if not point_set:
        return 0.0

    # 1. 最近距离相似度
    distances = [distance(reference_point, p) for p in point_set]
    min_dist = min(distances)
    dist_similarity = max(0, 1 - min_dist / max_distance)
    # 2. 点密度相似度
    # 计算参考点周围不同半径的点密度
    density_scores = []
    for radius in [1.0, 2.0, 5.0]:
        # 计算落在半径内的点数
        points_in_radius = sum(1 for d in distances if d <= radius)
        # 归一化
        if len(point_set) > 0:
            density_similarity = points_in_radius / len(point_set)
            density_scores.append(density_similarity)

    # 3. 方向分布相似度
    if len(point_set) > 1:
        # 计算从参考点到所有其他点的方向向量
        direction_vectors = [(p[0] - reference_point[0], p[1] - reference_point[1])
                             for p in point_set]
        # 标准化向量
        direction_vectors = [normalize_vector(v) for v in direction_vectors]
        # 计算方向覆盖度
        direction_coverage = compute_direction_coverage(direction_vectors)
    else:
        direction_coverage = 0

    # 计算综合相似度
    if density_scores:
        avg_density = sum(density_scores) / len(density_scores)
        # 权重组合三种相似度
        similarity = 0.5 * dist_similarity + 0.3 * avg_density + 0.2 * direction_coverage
    else:
        similarity = dist_similarity

    return similarity


def compute_line_set_similarity(reference_line, line_set, max_distance=10.0):
    """
    计算一条线与线集的相似度

    参数:
        reference_line: 参考线，列表包含多个点坐标
        line_set: 线集，列表包含多条线
        max_distance: 最大距离阈值，用于归一化

    返回:
        相似度分数 (0-1)
    """
    if not line_set:
        return 0.0

    # 1. 形状相似度 - 基于弗雷歇距离
    frechet_distances = [compute_frechet_distance(reference_line, line) for line in line_set]
    min_frechet_dist = min(frechet_distances)
    shape_similarity = max(0, 1 - min_frechet_dist / max_distance)

    # 2. 方向相似度
    direction_similarities = []

    # 参考线的方向：计算每个线段的方向，然后取平均
    ref_directions = []
    for i in range(len(reference_line) - 1):
        angle = compute_angle(reference_line[i], reference_line[i + 1])
        ref_directions.append(angle)

    if not ref_directions:
        direction_similarity = 0
    else:
        for line in line_set:
            if len(line) < 2:
                continue

            # 计算当前线的方向
            curr_directions = []
            for i in range(len(line) - 1):
                angle = compute_angle(line[i], line[i + 1])
                curr_directions.append(angle)

            if not curr_directions:
                continue

            # 计算方向差异
            # 对于每个参考方向，找到最接近的当前方向
            angle_diffs = []
            for ref_dir in ref_directions:
                min_diff = min((abs((ref_dir - curr_dir + math.pi) % (2 * math.pi) - math.pi)
                                for curr_dir in curr_directions), default=math.pi)
                angle_diffs.append(min_diff)

            # 取平均角度差异，转换为相似度
            avg_angle_diff = sum(angle_diffs) / len(angle_diffs)
            dir_sim = max(0, 1 - avg_angle_diff / math.pi)
            direction_similarities.append(dir_sim)

    # 3. 长度相似度
    length_similarities = []
    ref_length = sum(distance(reference_line[i], reference_line[i + 1])
                     for i in range(len(reference_line) - 1))

    for line in line_set:
        if len(line) < 2:
            continue

        curr_length = sum(distance(line[i], line[i + 1]) for i in range(len(line) - 1))

        # 计算长度比率
        if ref_length > 0 and curr_length > 0:
            ratio = min(ref_length, curr_length) / max(ref_length, curr_length)
            length_similarities.append(ratio)

    # 计算综合相似度
    avg_direction_similarity = sum(direction_similarities) / len(
        direction_similarities) if direction_similarities else 0
    avg_length_similarity = sum(length_similarities) / len(length_similarities) if length_similarities else 0

    # 权重组合三种相似度
    similarity = 0.5 * shape_similarity + 0.3 * avg_direction_similarity + 0.2 * avg_length_similarity

    return similarity


def compute_polygon_set_similarity(reference_polygon, polygon_set, max_distance=10.0):
    """
    计算一个多边形与多边形集的相似度

    参数:
        reference_polygon: 参考多边形，列表包含多个点坐标
        polygon_set: 多边形集，列表包含多个多边形
        max_distance: 最大距离阈值，用于归一化

    返回:
        相似度分数 (0-1)
    """
    if not polygon_set:
        return 0.0

    # 1. 形状相似度 - 基于豪斯多夫距离
    hausdorff_distances = [compute_hausdorff_distance(reference_polygon, poly) for poly in polygon_set]
    min_hausdorff_dist = min(hausdorff_distances)
    shape_similarity = max(0, 1 - min_hausdorff_dist / max_distance)

    # 2. 面积相似度
    area_similarities = []

    ref_shapely = ShapelyPolygon(reference_polygon)
    ref_area = ref_shapely.area

    for poly in polygon_set:
        poly_shapely = ShapelyPolygon(poly)
        poly_area = poly_shapely.area

        # 计算面积比率
        if ref_area > 0 and poly_area > 0:
            ratio = min(ref_area, poly_area) / max(ref_area, poly_area)
            area_similarities.append(ratio)

        # 计算IOUs
        try:
            iou = compute_iou(reference_polygon, poly)
            area_similarities.append(iou)
        except Exception as e:
            # 防止几何计算错误
            pass

    # 3. 轮廓相似度 - 周长比较
    contour_similarities = []

    ref_perimeter = ref_shapely.length

    for poly in polygon_set:
        poly_shapely = ShapelyPolygon(poly)
        poly_perimeter = poly_shapely.length

        # 计算周长比率
        if ref_perimeter > 0 and poly_perimeter > 0:
            ratio = min(ref_perimeter, poly_perimeter) / max(ref_perimeter, poly_perimeter)
            contour_similarities.append(ratio)

    # 计算综合相似度
    avg_area_similarity = sum(area_similarities) / len(area_similarities) if area_similarities else 0
    avg_contour_similarity = sum(contour_similarities) / len(contour_similarities) if contour_similarities else 0

    # 权重组合三种相似度
    similarity = 0.4 * shape_similarity + 0.4 * avg_area_similarity + 0.2 * avg_contour_similarity

    return similarity


def compute_overall_spatial_similarity(reference_points, reference_lines, reference_polygons,
                                       target_points, target_lines, target_polygons):
    """
    计算整体空间结构的相似度

    参数:
        reference_points, reference_lines, reference_polygons: 参考点、线、面集合
        target_points, target_lines, target_polygons: 目标点、线、面集合

    返回:
        空间结构相似度分数 (0-1)
    """
    # 构建空间关系图
    ref_graph, ref_stats = build_spatial_relation_graph(
        reference_points, reference_lines, reference_polygons
    )

    target_graph, target_stats = build_spatial_relation_graph(
        target_points, target_lines, target_polygons
    )

    # 比较空间关系统计
    relation_similarity = 0.0

    if sum(ref_stats.values()) > 0 and sum(target_stats.values()) > 0:
        similarities = []

        for rel_type in ref_stats:
            ref_count = ref_stats[rel_type]
            target_count = target_stats[rel_type]

            # 计算关系数量的比例
            if ref_count > 0 or target_count > 0:
                ratio = min(ref_count, target_count) / max(ref_count, target_count)
                similarities.append(ratio)

        if similarities:
            relation_similarity = sum(similarities) / len(similarities)

    # 拓扑相似度：比较节点度数分布
    ref_degrees = [d for _, d in ref_graph.degree()]
    target_degrees = [d for _, d in target_graph.degree()]

    topology_similarity = 0.0

    if ref_degrees and target_degrees:
        # 计算度数分布的EMD (Earth Mover's Distance)的简化版本
        # 这里使用直方图比较作为简化
        ref_hist, _ = np.histogram(ref_degrees, bins=10, range=(0, 10))
        target_hist, _ = np.histogram(target_degrees, bins=10, range=(0, 10))

        # 归一化直方图
        if np.sum(ref_hist) > 0:
            ref_hist = ref_hist / np.sum(ref_hist)
        if np.sum(target_hist) > 0:
            target_hist = target_hist / np.sum(target_hist)

        # 计算直方图相似度 (1 - 欧几里得距离/sqrt(2))
        hist_distance = np.sqrt(np.sum((ref_hist - target_hist) ** 2))
        topology_similarity = max(0, 1 - hist_distance / np.sqrt(2))

    # 密度相似度：比较节点密度
    ref_density = len(ref_graph.edges()) / len(ref_graph.nodes()) if len(ref_graph.nodes()) > 0 else 0
    target_density = len(target_graph.edges()) / len(target_graph.nodes()) if len(target_graph.nodes()) > 0 else 0

    density_similarity = 0.0

    if ref_density > 0 or target_density > 0:
        density_similarity = min(ref_density, target_density) / max(ref_density, target_density)

    # 综合相似度
    overall_similarity = 0.4 * relation_similarity + 0.4 * topology_similarity + 0.2 * density_similarity

    return overall_similarity


def compute_comprehensive_similarity(reference_points, reference_lines, reference_polygons,
                                     target_points, target_lines, target_polygons):
    """
    计算参考场景与目标场景的全面相似度

    参数:
        reference_points, reference_lines, reference_polygons: 参考点、线、面集合
        target_points, target_lines, target_polygons: 目标点、线、面集合

    返回:
        包含各层次相似度的字典
    """
    # 1. 点集相似度
    point_similarities = []
    for p in reference_points:
        sim = compute_point_set_similarity(p, target_points)
        point_similarities.append(sim)

    # 2. 线集相似度
    line_similarities = []
    for l in reference_lines:
        sim = compute_line_set_similarity(l, target_lines)
        line_similarities.append(sim)

    # 3. 多边形集相似度
    polygon_similarities = []
    for poly in reference_polygons:
        sim = compute_polygon_set_similarity(poly, target_polygons)
        polygon_similarities.append(sim)

    # 4. 整体空间结构相似度
    spatial_structure_similarity = compute_overall_spatial_similarity(
        reference_points, reference_lines, reference_polygons,
        target_points, target_lines, target_polygons
    )

    # 聚合各类相似度
    avg_point_similarity = sum(point_similarities) / len(point_similarities) if point_similarities else 0
    avg_line_similarity = sum(line_similarities) / len(line_similarities) if line_similarities else 0
    avg_polygon_similarity = sum(polygon_similarities) / len(polygon_similarities) if polygon_similarities else 0

    # 计算综合相似度（权重可调整）
    comprehensive_similarity = (
            0.2 * avg_point_similarity +
            0.3 * avg_line_similarity +
            0.3 * avg_polygon_similarity +
            0.2 * spatial_structure_similarity
    )

    return {
        'point_set_similarity': avg_point_similarity,
        'line_set_similarity': avg_line_similarity,
        'polygon_set_similarity': avg_polygon_similarity,
        'spatial_structure_similarity': spatial_structure_similarity,
        'comprehensive_similarity': comprehensive_similarity
    }


# ==============================
# 7. 神经网络模型训练函数
# ==============================

def train_model(model, train_loader, val_loader, epochs=20, lr=0.001):
    """
    训练形状相似度模型

    参数:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        lr: 学习率

    返回:
        训练历史记录
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            # 确保数据在正确的设备上
            batch1, batch2, similarity_target = (
                batch[0].to(device), batch[1].to(device), batch[2].to(device)
            )

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            similarity_pred = model(batch1, batch2)

            # 计算损失
            loss = criterion(
                similarity_pred['overall_similarity'],
                similarity_target['overall'].unsqueeze(1)
            )

            # 加上对各部分相似度的损失
            loss += criterion(
                similarity_pred['point_similarity'],
                similarity_target['point'].unsqueeze(1)
            )

            loss += criterion(
                similarity_pred['line_similarity'],
                similarity_target['line'].unsqueeze(1)
            )

            loss += criterion(
                similarity_pred['polygon_similarity'],
                similarity_target['polygon'].unsqueeze(1)
            )

            # 反向传播
            loss.backward()

            # 参数更新
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch1, batch2, similarity_target = (
                    batch[0].to(device), batch[1].to(device), batch[2].to(device)
                )

                similarity_pred = model(batch1, batch2)

                # 计算损失
                loss = criterion(
                    similarity_pred['overall_similarity'],
                    similarity_target['overall'].unsqueeze(1)
                )

                loss += criterion(
                    similarity_pred['point_similarity'],
                    similarity_target['point'].unsqueeze(1)
                )

                loss += criterion(
                    similarity_pred['line_similarity'],
                    similarity_target['line'].unsqueeze(1)
                )

                loss += criterion(
                    similarity_pred['polygon_similarity'],
                    similarity_target['polygon'].unsqueeze(1)
                )

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    return {'train_losses': train_losses, 'val_losses': val_losses}


# ==============================
# 8. 演示和可视化函数
# ==============================

def generate_random_shape_data(num_points=5, num_lines=3, num_polygons=2):
    """
    生成随机的形状数据用于演示

    参数:
        num_points: 点的数量
        num_lines: 线的数量
        num_polygons: 多边形的数量

    返回:
        点、线、面的集合
    """
    # 生成随机点
    points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_points)]

    # 生成随机线
    lines = []
    for _ in range(num_lines):
        num_vertices = random.randint(2, 6)
        line = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_vertices)]
        lines.append(line)

    # 生成随机多边形
    polygons = []
    for _ in range(num_polygons):
        # 生成简单的凸多边形
        center_x, center_y = random.uniform(20, 80), random.uniform(20, 80)
        num_vertices = random.randint(3, 8)

        # 生成极坐标角度和距离
        angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(num_vertices)])
        distances = [random.uniform(5, 15) for _ in range(num_vertices)]

        # 转换为笛卡尔坐标
        vertices = [(center_x + distances[i] * math.cos(angles[i]),
                     center_y + distances[i] * math.sin(angles[i]))
                    for i in range(num_vertices)]

        polygons.append(vertices)

    return points, lines, polygons


def apply_transformation(points, lines, polygons, translation=(0, 0), scale=1.0, noise_level=0.0):
    """
    对形状数据应用变换

    参数:
        points, lines, polygons: 形状数据
        translation: 平移向量
        scale: 缩放因子
        noise_level: 噪声水平

    返回:
        变换后的形状数据
    """
    # 应用变换到点
    transformed_points = []
    for p in points:
        # 缩放和平移
        new_x = p[0] * scale + translation[0]
        new_y = p[1] * scale + translation[1]

        # 添加随机噪声
        if noise_level > 0:
            new_x += random.uniform(-noise_level, noise_level)
            new_y += random.uniform(-noise_level, noise_level)

        transformed_points.append((new_x, new_y))

    # 应用变换到线
    transformed_lines = []
    for line in lines:
        # 随机决定是否保留此线（模拟数据丢失）
        if random.random() < 0.9:  # 90%的概率保留
            transformed_line = []
            for p in line:
                # 缩放和平移
                new_x = p[0] * scale + translation[0]
                new_y = p[1] * scale + translation[1]

                # 添加随机噪声
                if noise_level > 0:
                    new_x += random.uniform(-noise_level, noise_level)
                    new_y += random.uniform(-noise_level, noise_level)

                transformed_line.append((new_x, new_y))

            # 随机决定是否简化线（删除一些点）
            if len(transformed_line) > 2 and random.random() < 0.3:  # 30%的概率简化
                # 保留起点和终点，随机删除一些中间点
                to_keep = set([0, len(transformed_line) - 1])  # 确保保留起点和终点
                num_to_keep = max(2, int(len(transformed_line) * 0.7))  # 至少保留2个点
                while len(to_keep) < num_to_keep:
                    to_keep.add(random.randint(1, len(transformed_line) - 2))

                transformed_line = [transformed_line[i] for i in sorted(to_keep)]

            transformed_lines.append(transformed_line)

    # 应用变换到多边形
    transformed_polygons = []
    for poly in polygons:
        # 随机决定是否保留此多边形（模拟数据丢失）
        if random.random() < 0.9:  # 90%的概率保留
            transformed_poly = []
            for p in poly:
                # 缩放和平移
                new_x = p[0] * scale + translation[0]
                new_y = p[1] * scale + translation[1]

                # 添加随机噪声
                if noise_level > 0:
                    new_x += random.uniform(-noise_level, noise_level)
                    new_y += random.uniform(-noise_level, noise_level)

                transformed_poly.append((new_x, new_y))

            # 随机决定是否简化多边形（删除一些顶点）
            if len(transformed_poly) > 3 and random.random() < 0.3:  # 30%的概率简化
                # 随机删除一些顶点，但至少保留3个顶点
                num_to_keep = max(3, int(len(transformed_poly) * 0.7))
                to_keep = set(random.sample(range(len(transformed_poly)), num_to_keep))

                transformed_poly = [transformed_poly[i] for i in sorted(to_keep)]

            transformed_polygons.append(transformed_poly)

    # 随机添加一些新的点、线、多边形（模拟新增数据）
    if random.random() < 0.3:  # 30%的概率添加新数据
        # 添加新点
        num_new_points = random.randint(0, 2)
        for _ in range(num_new_points):
            new_x = random.uniform(min(p[0] for p in transformed_points), max(p[0] for p in transformed_points))
            new_y = random.uniform(min(p[1] for p in transformed_points), max(p[1] for p in transformed_points))
            transformed_points.append((new_x, new_y))

        # 添加新线
        num_new_lines = random.randint(0, 1)
        for _ in range(num_new_lines):
            num_vertices = random.randint(2, 4)
            center_x = random.uniform(min(p[0] for p in transformed_points), max(p[0] for p in transformed_points))
            center_y = random.uniform(min(p[1] for p in transformed_points), max(p[1] for p in transformed_points))

            new_line = [(center_x + random.uniform(-10, 10), center_y + random.uniform(-10, 10))
                        for _ in range(num_vertices)]
            transformed_lines.append(new_line)

    return transformed_points, transformed_lines, transformed_polygons


def visualize_shapes(points, lines, polygons, title="Shape Visualization"):
    """可视化形状数据"""
    plt.figure(figsize=(10, 8))

    # 绘制多边形
    for poly in polygons:
        polygon = Polygon(poly, alpha=0.5, facecolor='lightblue', edgecolor='blue', linewidth=2)
        plt.gca().add_patch(polygon)

    # 绘制线
    for line in lines:
        xs, ys = zip(*line)
        plt.plot(xs, ys, 'g-', linewidth=2)

    # 绘制点
    xs, ys = zip(*points) if points else ([], [])
    plt.scatter(xs, ys, c='red', s=50)

    plt.title(title)
    plt.grid(True)
    plt.axis('equal')

    return plt.gca()


def visualize_similarity_comparison(ref_points, ref_lines, ref_polygons,
                                    target_points, target_lines, target_polygons,
                                    similarity_scores):
    """可视化相似度比较"""
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))

    # 绘制参考形状
    ax1 = axs[0]

    # 绘制多边形
    for poly in ref_polygons:
        polygon = Polygon(poly, alpha=0.5, facecolor='lightblue', edgecolor='blue', linewidth=2)
        ax1.add_patch(polygon)

    # 绘制线
    for line in ref_lines:
        xs, ys = zip(*line)
        ax1.plot(xs, ys, 'g-', linewidth=2)

    # 绘制点
    xs, ys = zip(*ref_points) if ref_points else ([], [])
    ax1.scatter(xs, ys, c='red', s=50)

    ax1.set_title("参考形状")
    ax1.grid(True)
    ax1.axis('equal')

    # 绘制目标形状
    ax2 = axs[1]

    # 绘制多边形
    for poly in target_polygons:
        polygon = Polygon(poly, alpha=0.5, facecolor='lightblue', edgecolor='blue', linewidth=2)
        ax2.add_patch(polygon)

    # 绘制线
    for line in target_lines:
        xs, ys = zip(*line)
        ax2.plot(xs, ys, 'g-', linewidth=2)

    # 绘制点
    xs, ys = zip(*target_points) if target_points else ([], [])
    ax2.scatter(xs, ys, c='red', s=50)

    ax2.set_title("目标形状")
    ax2.grid(True)
    ax2.axis('equal')

    # 添加相似度信息
    info_text = "\n".join([
        f"点集相似度: {similarity_scores['point_set_similarity']:.4f}",
        f"线集相似度: {similarity_scores['line_set_similarity']:.4f}",
        f"多边形集相似度: {similarity_scores['polygon_set_similarity']:.4f}",
        f"空间结构相似度: {similarity_scores['spatial_structure_similarity']:.4f}",
        f"综合相似度: {similarity_scores['comprehensive_similarity']:.4f}"
    ])

    fig.text(0.5, 0.01, info_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


def create_synthetic_dataset(num_samples=100, noise_levels=[0.1, 0.5, 1.0, 2.0, 5.0]):
    """
    创建合成数据集，用于训练和测试

    参数:
        num_samples: 生成的样本数量
        noise_levels: 噪声水平列表

    返回:
        数据集字典
    """
    dataset = []

    for _ in range(num_samples):
        # 生成原始形状
        original_points, original_lines, original_polygons = generate_random_shape_data()

        # 随机选择一种变换类型
        transform_type = random.choice(['translation', 'scale', 'noise', 'mixed'])

        # 随机选择噪声水平
        noise_level = random.choice(noise_levels)

        # 根据变换类型应用不同变换
        if transform_type == 'translation':
            # 平移变换
            translation = (random.uniform(-10, 10), random.uniform(-10, 10))
            transformed_points, transformed_lines, transformed_polygons = apply_transformation(
                original_points, original_lines, original_polygons,
                translation=translation, scale=1.0, noise_level=noise_level
            )

        elif transform_type == 'scale':
            # 缩放变换
            scale = random.uniform(0.5, 1.5)
            transformed_points, transformed_lines, transformed_polygons = apply_transformation(
                original_points, original_lines, original_polygons,
                translation=(0, 0), scale=scale, noise_level=noise_level
            )

        elif transform_type == 'noise':
            # 仅添加噪声
            transformed_points, transformed_lines, transformed_polygons = apply_transformation(
                original_points, original_lines, original_polygons,
                translation=(0, 0), scale=1.0, noise_level=noise_level
            )

        else:  # 'mixed'
            # 混合变换
            translation = (random.uniform(-10, 10), random.uniform(-10, 10))
            scale = random.uniform(0.5, 1.5)
            transformed_points, transformed_lines, transformed_polygons = apply_transformation(
                original_points, original_lines, original_polygons,
                translation=translation, scale=scale, noise_level=noise_level
            )

            # 计算相似度
        similarity = compute_comprehensive_similarity(
            original_points, original_lines, original_polygons,
            transformed_points, transformed_lines, transformed_polygons
        )

        # 创建原始形状的图结构
        original_graph, _ = build_spatial_relation_graph(
            original_points, original_lines, original_polygons
        )

        # 创建变换后形状的图结构
        transformed_graph, _ = build_spatial_relation_graph(
            transformed_points, transformed_lines, transformed_polygons
        )

        # 转换为PyG数据对象
        original_pyg = convert_nx_to_pyg(original_graph)
        transformed_pyg = convert_nx_to_pyg(transformed_graph)

        # 添加到数据集
        dataset.append({
            'original': {
                'points': original_points,
                'lines': original_lines,
                'polygons': original_polygons,
                'graph': original_graph,
                'pyg_data': original_pyg
            },
            'transformed': {
                'points': transformed_points,
                'lines': transformed_lines,
                'polygons': transformed_polygons,
                'graph': transformed_graph,
                'pyg_data': transformed_pyg
            },
            'similarity': similarity,
            'transform_type': transform_type,
            'noise_level': noise_level
        })

        # 划分训练、验证和测试集
    random.shuffle(dataset)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
        'all': dataset
    }


def prepare_dataloader(dataset, batch_size=8):
    """
    准备数据加载器

    参数:
        dataset: 数据集
        batch_size: 批量大小

    返回:
        数据加载器
    """
    # 预处理数据集
    processed_dataset = []

    for sample in dataset:
        # 提取PyG数据
        original_pyg = sample['original']['pyg_data']
        transformed_pyg = sample['transformed']['pyg_data']

        # 提取相似度标签
        similarity = {
            'point': torch.tensor(sample['similarity']['point_set_similarity'], dtype=torch.float),
            'line': torch.tensor(sample['similarity']['line_set_similarity'], dtype=torch.float),
            'polygon': torch.tensor(sample['similarity']['polygon_set_similarity'], dtype=torch.float),
            'structure': torch.tensor(sample['similarity']['spatial_structure_similarity'], dtype=torch.float),
            'overall': torch.tensor(sample['similarity']['comprehensive_similarity'], dtype=torch.float)
        }

        processed_dataset.append((original_pyg, transformed_pyg, similarity))

    # 创建数据加载器
    dataloader = DataLoader(processed_dataset, batch_size=batch_size, shuffle=True)

    return dataloader


# ==============================
# 9. 主程序：演示相似度计算
# ==============================

def main():
    # 示例：生成随机形状数据
    print("生成随机形状数据...")
    ref_points, ref_lines, ref_polygons = generate_random_shape_data(
        num_points=8, num_lines=4, num_polygons=3
    )

    # 生成变换后的形状
    print("生成变换后的形状...")
    transformed_points, transformed_lines, transformed_polygons = apply_transformation(
        ref_points, ref_lines, ref_polygons,
        translation=(5, 5), scale=0.8, noise_level=2.0
    )

    # 计算综合相似度
    print("计算综合相似度...")
    similarity = compute_comprehensive_similarity(
        ref_points, ref_lines, ref_polygons,
        transformed_points, transformed_lines, transformed_polygons
    )

    # 打印相似度结果
    print("\n相似度计算结果:")
    print(f"点集相似度: {similarity['point_set_similarity']:.4f}")
    print(f"线集相似度: {similarity['line_set_similarity']:.4f}")
    print(f"多边形集相似度: {similarity['polygon_set_similarity']:.4f}")
    print(f"空间结构相似度: {similarity['spatial_structure_similarity']:.4f}")
    print(f"综合相似度: {similarity['comprehensive_similarity']:.4f}")

    # 可视化比较
    print("\n可视化比较...")
    fig = visualize_similarity_comparison(
        ref_points, ref_lines, ref_polygons,
        transformed_points, transformed_lines, transformed_polygons,
        similarity
    )
    plt.savefig('similarity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("可视化结果已保存为 'similarity_comparison.png'")

    # 构建空间关系图
    print("\n构建空间关系图...")
    ref_graph, ref_stats = build_spatial_relation_graph(
        ref_points, ref_lines, ref_polygons
    )

    transformed_graph, transformed_stats = build_spatial_relation_graph(
        transformed_points, transformed_lines, transformed_polygons
    )

    # 打印图统计信息
    print("\n原始形状图统计:")
    for rel_type, count in ref_stats.items():
        print(f"  {rel_type}: {count}")

    print("\n变换后形状图统计:")
    for rel_type, count in transformed_stats.items():
        print(f"  {rel_type}: {count}")

    # 转换为PyG数据对象
    print("\n转换为PyG数据对象...")
    ref_pyg = convert_nx_to_pyg(ref_graph)
    transformed_pyg = convert_nx_to_pyg(transformed_graph)

    print("原始形状PyG数据:")
    print(f"  节点特征形状: {ref_pyg.x.shape}")
    print(f"  边索引形状: {ref_pyg.edge_index.shape}")
    print(f"  边特征形状: {ref_pyg.edge_attr.shape}")

    # 示例：创建合成数据集
    print("\n创建合成数据集(简化版)...")
    mini_dataset = create_synthetic_dataset(num_samples=10)

    print(f"训练集大小: {len(mini_dataset['train'])}")
    print(f"验证集大小: {len(mini_dataset['val'])}")
    print(f"测试集大小: {len(mini_dataset['test'])}")

    # 创建模型
    print("\n创建形状相似度模型...")
    model = ShapeSimilarityModel()

    # 打印模型结构
    print(model)

    print("\n完成演示！")


# ==============================
# 10. 实际应用示例：多尺度地图相似度计算
# ==============================

def compute_map_similarity(reference_map, target_map):
    """
    计算两个地图之间的相似度

    参数:
        reference_map: 参考地图，包含点、线、面集合
        target_map: 目标地图，包含点、线、面集合

    返回:
        相似度得分
    """
    # 提取地图要素
    ref_points = reference_map['points']
    ref_lines = reference_map['lines']
    ref_polygons = reference_map['polygons']

    target_points = target_map['points']
    target_lines = target_map['lines']
    target_polygons = target_map['polygons']

    # 计算综合相似度
    similarity = compute_comprehensive_similarity(
        ref_points, ref_lines, ref_polygons,
        target_points, target_lines, target_polygons
    )

    return similarity


def demonstrate_map_scale_similarity():
    """演示多尺度地图相似度计算"""
    # 1. 模拟不同尺度的地图数据

    # 1.1 大比例尺地图（详细）
    large_scale_map = {
        'scale': '1:10000',
        'points': [(25, 25), (75, 25), (25, 75), (75, 75), (50, 50)],
        'lines': [
            [(10, 10), (90, 10), (90, 30), (10, 30), (10, 10)],  # 外部边界路
            [(25, 25), (75, 25), (75, 75), (25, 75), (25, 25)],  # 内部边界路
            [(25, 25), (75, 75)],  # 对角线路1
            [(25, 75), (75, 25)]  # 对角线路2
        ],
        'polygons': [
            [(30, 30), (45, 30), (45, 45), (30, 45)],  # 小建筑1
            [(55, 30), (70, 30), (70, 45), (55, 45)],  # 小建筑2
            [(30, 55), (45, 55), (45, 70), (30, 70)],  # 小建筑3
            [(55, 55), (70, 55), (70, 70), (55, 70)]  # 小建筑4
        ]
    }

    # 1.2 中比例尺地图（简化）
    medium_scale_map = {
        'scale': '1:25000',
        'points': [(25, 25), (75, 25), (25, 75), (75, 75)],  # 减少了中心点
        'lines': [
            [(10, 10), (90, 10), (90, 30), (10, 30), (10, 10)],  # 保持外部边界路
            [(25, 25), (75, 25), (75, 75), (25, 75), (25, 25)],  # 保持内部边界路
            [(25, 25), (75, 75)]  # 仅保留一条对角线路
        ],
        'polygons': [
            [(30, 30), (45, 30), (45, 45), (30, 45)],  # 保留建筑1
            [(55, 30), (70, 30), (70, 45), (55, 45)],  # 保留建筑2
            # 合并建筑3和4为一个大建筑
            [(30, 55), (70, 55), (70, 70), (30, 70)]
        ]
    }

    # 1.3 小比例尺地图（高度简化）
    small_scale_map = {
        'scale': '1:50000',
        'points': [(50, 50)],  # 仅保留中心点
        'lines': [
            [(10, 10), (90, 10), (90, 30), (10, 30), (10, 10)],  # 保持外部边界路
            [(25, 25), (75, 25), (75, 75), (25, 75), (25, 25)]  # 保持内部边界路
        ],
        'polygons': [
            # 所有建筑合并为一个
            [(30, 30), (70, 30), (70, 70), (30, 70)]
        ]
    }

    # 2. 计算不同尺度地图间的相似度

    print("\n多尺度地图相似度计算演示")
    print("=" * 50)

    # 2.1 大比例尺与中比例尺比较
    large_medium_similarity = compute_map_similarity(large_scale_map, medium_scale_map)

    print("\n大比例尺 (1:10000) 与中比例尺 (1:25000) 地图相似度:")
    print(f"点集相似度: {large_medium_similarity['point_set_similarity']:.4f}")
    print(f"线集相似度: {large_medium_similarity['line_set_similarity']:.4f}")
    print(f"多边形集相似度: {large_medium_similarity['polygon_set_similarity']:.4f}")
    print(f"空间结构相似度: {large_medium_similarity['spatial_structure_similarity']:.4f}")
    print(f"综合相似度: {large_medium_similarity['comprehensive_similarity']:.4f}")

    # 可视化比较
    fig1 = visualize_similarity_comparison(
        large_scale_map['points'], large_scale_map['lines'], large_scale_map['polygons'],
        medium_scale_map['points'], medium_scale_map['lines'], medium_scale_map['polygons'],
        large_medium_similarity
    )
    plt.savefig('large_medium_similarity.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # 2.2 大比例尺与小比例尺比较
    large_small_similarity = compute_map_similarity(large_scale_map, small_scale_map)

    print("\n大比例尺 (1:10000) 与小比例尺 (1:50000) 地图相似度:")
    print(f"点集相似度: {large_small_similarity['point_set_similarity']:.4f}")
    print(f"线集相似度: {large_small_similarity['line_set_similarity']:.4f}")
    print(f"多边形集相似度: {large_small_similarity['polygon_set_similarity']:.4f}")
    print(f"空间结构相似度: {large_small_similarity['spatial_structure_similarity']:.4f}")
    print(f"综合相似度: {large_small_similarity['comprehensive_similarity']:.4f}")

    # 可视化比较
    fig2 = visualize_similarity_comparison(
        large_scale_map['points'], large_scale_map['lines'], large_scale_map['polygons'],
        small_scale_map['points'], small_scale_map['lines'], small_scale_map['polygons'],
        large_small_similarity
    )
    plt.savefig('large_small_similarity.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # 2.3 中比例尺与小比例尺比较
    medium_small_similarity = compute_map_similarity(medium_scale_map, small_scale_map)

    print("\n中比例尺 (1:25000) 与小比例尺 (1:50000) 地图相似度:")
    print(f"点集相似度: {medium_small_similarity['point_set_similarity']:.4f}")
    print(f"线集相似度: {medium_small_similarity['line_set_similarity']:.4f}")
    print(f"多边形集相似度: {medium_small_similarity['polygon_set_similarity']:.4f}")
    print(f"空间结构相似度: {medium_small_similarity['spatial_structure_similarity']:.4f}")
    print(f"综合相似度: {medium_small_similarity['comprehensive_similarity']:.4f}")

    # 可视化比较
    fig3 = visualize_similarity_comparison(
        medium_scale_map['points'], medium_scale_map['lines'], medium_scale_map['polygons'],
        small_scale_map['points'], small_scale_map['lines'], small_scale_map['polygons'],
        medium_small_similarity
    )
    plt.savefig('medium_small_similarity.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)

    print("\n相似度比较可视化已保存为图片文件")

    # 3. 构建和比较空间关系图

    # 3.1 构建空间关系图
    large_graph, large_stats = build_spatial_relation_graph(
        large_scale_map['points'], large_scale_map['lines'], large_scale_map['polygons']
    )

    medium_graph, medium_stats = build_spatial_relation_graph(
        medium_scale_map['points'], medium_scale_map['lines'], medium_scale_map['polygons']
    )

    small_graph, small_stats = build_spatial_relation_graph(
        small_scale_map['points'], small_scale_map['lines'], small_scale_map['polygons']
    )

    # 3.2 打印图统计信息
    print("\n空间关系图统计:")
    print("-" * 30)

    print("\n大比例尺地图关系统计:")
    for rel_type, count in large_stats.items():
        print(f"  {rel_type}: {count}")

    print("\n中比例尺地图关系统计:")
    for rel_type, count in medium_stats.items():
        print(f"  {rel_type}: {count}")

    print("\n小比例尺地图关系统计:")
    for rel_type, count in small_stats.items():
        print(f"  {rel_type}: {count}")

    print("\n多尺度地图相似度计算演示完成")


if __name__ == "__main__":
    # 运行主程序
    main()

    # 演示多尺度地图相似度计算
    demonstrate_map_scale_similarity()
