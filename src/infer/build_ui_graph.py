import json
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class RelationType(Enum):
    """UI component relationships"""
    PARENT_CHILD = "parent_child"  # Containment
    SIBLING = "sibling"  # Same level
    ABOVE = "above"  # Vertical ordering
    BELOW = "below"
    LEFT_OF = "left_of"  # Horizontal ordering
    RIGHT_OF = "right_of"


@dataclass
class Edge:
    """Graph edge between two components"""
    source_id: int
    target_id: int
    relation_type: RelationType
    confidence: float = 1.0


class UIGraphBuilder:
    """Builds hierarchical graph from detected components"""

    def __init__(self,
                 iou_threshold: float = 0.8,
                 vertical_threshold: float = 20,
                 horizontal_threshold: float = 20):
        """Graf kurucusunu içerme (containment) ve hizalama eşikleriyle başlatır.

        Girdi:  iou_threshold — içerme (ebeveyn-çocuk) kararı için IoU eşiği;
                vertical_threshold — dikey hizalama için piksel eşiği;
                horizontal_threshold — yatay hizalama için piksel eşiği
        Çıktı:  yok (eşikler nesne alanlarına atanır yan etkisi)
        """
        self.iou_threshold = iou_threshold
        self.v_thresh = vertical_threshold
        self.h_thresh = horizontal_threshold

    def build_graph(self, detections: List[Dict]) -> Dict:
        """Tespit edilen bileşenlerden tam UI grafını kurar: içerme, hiyerarşi, yerleşim grupları ve uzamsal ilişkiler.

        Girdi:  detections — YOLO tespitlerinden gelen bileşen sözlükleri listesi
        Çıktı:  {'nodes': bileşenler, 'edges': ilişkiler, 'hierarchy': ebeveyn-çocuk ağacı} sözlüğü
        """
        nodes = detections.copy()
        edges = []

        # 1. Find parent-child relationships (containment)
        containment_edges = self._find_containment_relations(nodes)
        edges.extend(containment_edges)

        # 2. Build hierarchy tree
        hierarchy = self._build_hierarchy_tree(nodes, containment_edges)

        # 3. Layout inference — group root nodes into Row/Column containers
        hierarchy = self._infer_layout_groups(hierarchy)

        # 4. Find spatial relationships (for same-level components)
        spatial_edges = self._find_spatial_relations(nodes, hierarchy)
        edges.extend(spatial_edges)

        return {
            'nodes': nodes,
            'edges': [self._edge_to_dict(e) for e in edges],
            'hierarchy': hierarchy
        }

    # ── Layout Inference ───────────────────────────────────────

    def _infer_layout_groups(self, hierarchy: Dict) -> Dict:
        """Kök düzeyindeki düğümleri y/x hizalamasına göre kümeleyip sentetik SyntheticRow/SyntheticColumn kapsayıcılarında gruplar.

        Girdi:  hierarchy — 'roots' listesini içeren hiyerarşi sözlüğü
        Çıktı:  kökleri sentetik kapsayıcılarla sarılmış güncellenmiş hiyerarşi sözlüğü
        """
        roots = hierarchy['roots']
        if len(roots) < 2:
            return hierarchy

        # Step 1: Sort roots by y-center
        for r in roots:
            bbox = r['bbox']
            r['_cy'] = (bbox[1] + bbox[3]) / 2
            r['_cx'] = (bbox[0] + bbox[2]) / 2
            r['_h'] = bbox[3] - bbox[1]
        roots.sort(key=lambda r: r['_cy'])

        # Step 2: Cluster into horizontal rows (similar y-center)
        avg_height = np.mean([r['_h'] for r in roots]) if roots else 50
        y_threshold = avg_height * 0.6
        rows = []
        current_row = [roots[0]]

        for i in range(1, len(roots)):
            if abs(roots[i]['_cy'] - current_row[-1]['_cy']) < y_threshold:
                current_row.append(roots[i])
            else:
                rows.append(current_row)
                current_row = [roots[i]]
        rows.append(current_row)

        # Step 2.5: Merge consecutive rows with matching x-alignment
        rows = self._merge_vertically_aligned_rows(rows, avg_height)

        # Step 3: Build final structure from rows
        new_roots = []
        synthetic_id = -1

        for row in rows:
            if len(row) == 1:
                node = row[0]
                self._clean_temp_fields(node)
                new_roots.append(node)
            else:
                row.sort(key=lambda r: r['_cx'])

                # Check if this row contains merged vertical groups
                has_vertical = any(isinstance(r.get('_vgroup'), list) for r in row)

                if has_vertical:
                    # Build Row with Column sub-groups
                    row_children = []
                    for item in row:
                        vgroup = item.pop('_vgroup', None)
                        if vgroup and len(vgroup) > 1:
                            # Vertical sub-group → SyntheticColumn
                            vgroup.sort(key=lambda r: r['_cy'])
                            for v in vgroup:
                                self._clean_temp_fields(v)
                            col_node = self._make_synthetic_node(
                                synthetic_id, 'SyntheticColumn', vgroup
                            )
                            synthetic_id -= 1
                            row_children.append(col_node)
                        else:
                            self._clean_temp_fields(item)
                            row_children.append(item)

                    row_node = self._make_synthetic_node(
                        synthetic_id, 'SyntheticRow', row_children
                    )
                    synthetic_id -= 1
                    new_roots.append(row_node)
                else:
                    # Simple horizontal row
                    for item in row:
                        self._clean_temp_fields(item)
                    row_node = self._make_synthetic_node(
                        synthetic_id, 'SyntheticRow', row
                    )
                    synthetic_id -= 1
                    new_roots.append(row_node)

        # Step 4: Wrap everything in a root SyntheticColumn
        if len(new_roots) > 1:
            root_column = self._make_synthetic_node(
                synthetic_id, 'SyntheticColumn', new_roots
            )
            hierarchy['roots'] = [root_column]
        else:
            hierarchy['roots'] = new_roots

        return hierarchy

    def _merge_vertically_aligned_rows(self, rows: List[List[Dict]],
                                         avg_height: float) -> List[List[Dict]]:
        """Öğeleri dikeyde hizalanan (x-merkezleri eşleşen) ardışık satırları dikey gruplar (_vgroup) hâlinde birleştirir.

        Örnek: ["Reward Points", "Travel Trips", "Bucket List"] (satır A) ile
               ["360", "238", "473"] (satır B) hizalıysa tek satıra birleşir:
               [("Reward Points","360"), ("Travel Trips","238"), ("Bucket List","473")]

        Girdi:  rows — kök düğüm satırları (liste listesi); avg_height — x hizalama toleransı için ortalama yükseklik
        Çıktı:  birleştirilmiş satır listesi (_vgroup işaretli temsilci düğümlerle)
        """
        if len(rows) < 2:
            return rows

        x_threshold = avg_height * 1.5  # x-alignment tolerance
        merged = []
        i = 0

        while i < len(rows):
            if i + 1 < len(rows):
                row_a = rows[i]
                row_b = rows[i + 1]

                # Only try merge if both rows have same number of elements (2+)
                if len(row_a) >= 2 and len(row_a) == len(row_b):
                    # Sort both by x-center
                    row_a_sorted = sorted(row_a, key=lambda n: n['_cx'])
                    row_b_sorted = sorted(row_b, key=lambda n: n['_cx'])

                    # Check if x-centers align pairwise
                    aligned = True
                    for a, b in zip(row_a_sorted, row_b_sorted):
                        if abs(a['_cx'] - b['_cx']) > x_threshold:
                            aligned = False
                            break

                    if aligned:
                        # Merge: create virtual groups with _vgroup marker
                        merged_row = []
                        for a, b in zip(row_a_sorted, row_b_sorted):
                            # Use 'a' as the representative, attach vertical group
                            a['_vgroup'] = [a.copy(), b]
                            merged_row.append(a)
                        merged.append(merged_row)
                        i += 2  # skip both rows
                        continue

            merged.append(rows[i])
            i += 1

        return merged

    def _cluster_by_x(self, nodes: List[Dict], threshold: float) -> List[List[Dict]]:
        """Düğümleri x-merkez konumlarına göre kümeler (grup ortalamasıyla karşılaştırarak).

        Girdi:  nodes — _cx alanı hesaplanmış düğüm listesi; threshold — kümeleme toleransı (piksel)
        Çıktı:  x eksenine göre gruplanmış düğüm listelerinin listesi
        """
        if not nodes:
            return []
        nodes_sorted = sorted(nodes, key=lambda n: n['_cx'])
        groups = [[nodes_sorted[0]]]
        for i in range(1, len(nodes_sorted)):
            # Compare with the average x of current group
            group_avg_x = np.mean([n['_cx'] for n in groups[-1]])
            if abs(nodes_sorted[i]['_cx'] - group_avg_x) < threshold:
                groups[-1].append(nodes_sorted[i])
            else:
                groups.append([nodes_sorted[i]])
        return groups

    def _make_synthetic_node(self, node_id: int, class_name: str,
                              children: List[Dict]) -> Dict:
        """Çocukların kutularını saran bbox ile sentetik kapsayıcı düğüm (SyntheticRow/SyntheticColumn) üretir.

        Girdi:  node_id — negatif sentetik kimlik; class_name — 'SyntheticRow' veya 'SyntheticColumn';
                children — kapsayıcıya girecek çocuk düğümler
        Çıktı:  id, class, bbox ve children alanlı kapsayıcı düğüm sözlüğü
        """
        all_bboxes = []
        def collect(n):
            """Düğümün ve tüm alt ağacının bbox'larını all_bboxes listesinde toplar.

            Girdi:  n — bbox ve children alanları olabilen düğüm sözlüğü
            Çıktı:  yok (all_bboxes listesine ekleme yan etkisi)
            """
            if 'bbox' in n:
                all_bboxes.append(n['bbox'])
            for c in n.get('children', []):
                collect(c)
        for c in children:
            collect(c)

        if all_bboxes:
            bbox = [
                min(b[0] for b in all_bboxes),
                min(b[1] for b in all_bboxes),
                max(b[2] for b in all_bboxes),
                max(b[3] for b in all_bboxes),
            ]
        else:
            bbox = [0, 0, 0, 0]

        return {
            'id': node_id,
            'class': class_name,
            'class_name': class_name,
            'bbox': bbox,
            'children': children,
        }

    def _clean_temp_fields(self, node: Dict):
        """Yerleşim çıkarımı sırasında kullanılan geçici alanları (_cy, _cx, _h) düğümden siler.

        Girdi:  node — temizlenecek düğüm sözlüğü
        Çıktı:  yok (düğüm yerinde güncellenir yan etkisi)
        """
        node.pop('_cy', None)
        node.pop('_cx', None)
        node.pop('_h', None)

    def _find_containment_relations(self, nodes: List[Dict]) -> List[Edge]:
        """Sınırlayıcı kutu içermesine (containment) dayanarak ebeveyn-çocuk kenarlarını bulur.

        Girdi:  nodes — bbox alanlı bileşen düğümleri listesi
        Çıktı:  geçişli olanları elenmiş PARENT_CHILD türünde Edge listesi
        """
        edges = []

        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i == j:
                    continue

                # Check if node_i contains node_j
                if self._is_contained(node_j['bbox'], node_i['bbox']):
                    # node_i is parent of node_j
                    edges.append(Edge(
                        source_id=node_i['id'],
                        target_id=node_j['id'],
                        relation_type=RelationType.PARENT_CHILD,
                        confidence=1.0
                    ))

        # Filter transitive relations (keep only direct parents)
        edges = self._remove_transitive_containment(edges, nodes)

        return edges

    def _is_contained(self, inner_box: List[float], outer_box: List[float]) -> bool:
        """İç kutunun dış kutunun tamamen içinde olup olmadığını denetler (içerme testi).

        Girdi:  inner_box, outer_box — (x1, y1, x2, y2) biçiminde iki kutu
        Çıktı:  True/False — iç kutu dış kutunun içindeyse True
        """
        ix1, iy1, ix2, iy2 = inner_box
        ox1, oy1, ox2, oy2 = outer_box

        # Inner box must be completely inside outer box
        return (ox1 <= ix1 and iy1 >= oy1 and
                ix2 <= ox2 and iy2 <= oy2)

    def _remove_transitive_containment(self, edges: List[Edge], nodes: List[Dict]) -> List[Edge]:
        """Dolaylı (geçişli) ebeveyn-çocuk kenarlarını eleyip yalnızca doğrudan ebeveynleri bırakır.

        Girdi:  edges — PARENT_CHILD kenar listesi; nodes — bileşen düğümleri listesi
        Çıktı:  yalnızca doğrudan ebeveyn-çocuk ilişkilerini içeren Edge listesi
        """
        # Build adjacency list
        children = {}  # parent_id -> [child_ids]
        for edge in edges:
            if edge.source_id not in children:
                children[edge.source_id] = []
            children[edge.source_id].append(edge.target_id)

        # Find direct parents for each child
        direct_edges = []
        for edge in edges:
            parent_id = edge.source_id
            child_id = edge.target_id

            # Check if there's an intermediate parent
            has_intermediate = False
            if parent_id in children:
                for other_child in children[parent_id]:
                    if other_child != child_id and other_child in children:
                        if child_id in children[other_child]:
                            has_intermediate = True
                            break

            if not has_intermediate:
                direct_edges.append(edge)

        return direct_edges

    def _build_hierarchy_tree(self, nodes: List[Dict], containment_edges: List[Edge]) -> Dict:
        """İçerme kenarlarından kök düğümleri bulup özyinelemeli ebeveyn-çocuk ağacını kurar.

        Girdi:  nodes — bileşen düğümleri; containment_edges — PARENT_CHILD kenarları
        Çıktı:  {'roots': [ağaç düğümleri]} biçiminde hiyerarşi sözlüğü
        """
        # Create parent mapping
        parent_map = {}  # child_id -> parent_id
        for edge in containment_edges:
            parent_map[edge.target_id] = edge.source_id

        # Find root nodes (no parent)
        roots = []
        for node in nodes:
            if node['id'] not in parent_map:
                roots.append(node['id'])

        # Build tree recursively
        def build_subtree(node_id):
            """Verilen düğümden başlayarak alt ağacı özyinelemeli kurar.

            Girdi:  node_id — alt ağacın kökü olacak düğümün kimliği
            Çıktı:  id, class, bbox ve children alanlı ağaç düğümü sözlüğü
            """
            node = next(n for n in nodes if n['id'] == node_id)
            children_ids = [e.target_id for e in containment_edges if e.source_id == node_id]

            return {
                'id': node_id,
                'class': node['class_name'],
                'bbox': node['bbox'],
                'children': [build_subtree(cid) for cid in children_ids]
            }

        return {
            'roots': [build_subtree(rid) for rid in roots]
        }

    def _find_spatial_relations(self, nodes: List[Dict], hierarchy: Dict) -> List[Edge]:
        """Aynı ebeveyni paylaşan kardeş bileşenler arasında uzamsal (üst/alt/sol/sağ) kenarları bulur.

        Girdi:  nodes — bileşen düğümleri; hierarchy — ebeveyn eşlemesi çıkarılacak hiyerarşi ağacı
        Çıktı:  ABOVE/BELOW/LEFT_OF/RIGHT_OF türünde Edge listesi
        """
        edges = []

        # Get parent mapping
        parent_map = self._get_parent_mapping(hierarchy)

        # Group components by parent (siblings)
        siblings_by_parent = {}
        for node in nodes:
            parent_id = parent_map.get(node['id'], None)
            if parent_id not in siblings_by_parent:
                siblings_by_parent[parent_id] = []
            siblings_by_parent[parent_id].append(node)

        # Find spatial relations within sibling groups
        for parent_id, siblings in siblings_by_parent.items():
            if len(siblings) < 2:
                continue

            for i, node_i in enumerate(siblings):
                for j, node_j in enumerate(siblings):
                    if i >= j:
                        continue

                    relation = self._determine_spatial_relation(node_i, node_j)
                    if relation:
                        edges.append(Edge(
                            source_id=node_i['id'],
                            target_id=node_j['id'],
                            relation_type=relation,
                            confidence=0.9
                        ))

        return edges

    def _determine_spatial_relation(self, node1: Dict, node2: Dict) -> RelationType:
        """İki bileşenin merkez konumlarına ve hizalama eşiklerine göre uzamsal ilişkiyi belirler.

        Girdi:  node1, node2 — 'center' alanlı iki bileşen düğümü
        Çıktı:  RelationType (ABOVE/BELOW/LEFT_OF/RIGHT_OF) ya da hizalama yoksa None
        """
        x1c, y1c = node1['center']
        x2c, y2c = node2['center']

        dx = x2c - x1c
        dy = y2c - y1c

        # Vertical relationships
        if abs(dx) < self.h_thresh:
            if dy > 0:
                return RelationType.ABOVE  # node1 is above node2
            else:
                return RelationType.BELOW

        # Horizontal relationships
        if abs(dy) < self.v_thresh:
            if dx > 0:
                return RelationType.LEFT_OF  # node1 is left of node2
            else:
                return RelationType.RIGHT_OF

        return None

    def _get_parent_mapping(self, hierarchy: Dict) -> Dict[int, int]:
        """Hiyerarşi ağacını gezerek çocuk→ebeveyn kimlik eşlemesini çıkarır.

        Girdi:  hierarchy — 'roots' listesini içeren hiyerarşi sözlüğü
        Çıktı:  {çocuk_id: ebeveyn_id} biçiminde eşleme sözlüğü
        """
        parent_map = {}

        def traverse(node, parent_id=None):
            """Alt ağacı gezerek her çocuğu ebeveyniyle parent_map'e kaydeder.

            Girdi:  node — gezilecek düğüm; parent_id — düğümün ebeveyn kimliği (kökte None)
            Çıktı:  yok (parent_map sözlüğüne yazma yan etkisi)
            """
            if parent_id is not None:
                parent_map[node['id']] = parent_id
            for child in node.get('children', []):
                traverse(child, node['id'])

        for root in hierarchy['roots']:
            traverse(root)

        return parent_map

    def _edge_to_dict(self, edge: Edge) -> Dict:
        """Edge nesnesini JSON'a yazılabilir sözlüğe dönüştürür.

        Girdi:  edge — kaynak, hedef, ilişki türü ve güven değeri taşıyan Edge
        Çıktı:  source, target, relation, confidence anahtarlı sözlük
        """
        return {
            'source': edge.source_id,
            'target': edge.target_id,
            'relation': edge.relation_type.value,
            'confidence': edge.confidence
        }

    def visualize_graph(self, graph: Dict, output_path: str = "graph.json"):
        """Grafı görselleştirme için JSON dosyası olarak kaydeder.

        Girdi:  graph — build_graph çıktısı sözlük; output_path — hedef JSON dosya yolu
        Çıktı:  yok (diske JSON yazma ve konsol bildirimi yan etkisi)
        """
        with open(output_path, 'w') as f:
            json.dump(graph, f, indent=2)
        print(f"✓ Graph saved: {output_path}")


def main():
    """Graf kurucusunu kayıtlı tespit çıktısı üzerinde uçtan uca dener ve hiyerarşiyi yazdırır.

    Girdi:  yok (output/detections.json dosyasından okur)
    Çıktı:  yok (output/ui_graph.json kaydetme ve konsola özet yazdırma yan etkisi)
    """
    # Load detections from previous step
    with open("output/detections.json", 'r') as f:
        detections = json.load(f)

    # Build graph
    builder = UIGraphBuilder()
    graph = builder.build_graph(detections)

    print(f"\n🌳 Graph Statistics:")
    print(f"  Nodes: {len(graph['nodes'])}")
    print(f"  Edges: {len(graph['edges'])}")
    print(f"  Root components: {len(graph['hierarchy']['roots'])}")

    # Save graph
    builder.visualize_graph(graph, "output/ui_graph.json")

    # Print hierarchy
    print(f"\n📊 Hierarchy:")

    def print_tree(node, indent=0):
        """Hiyerarşi ağacını girintili biçimde konsola yazdırır.

        Girdi:  node — yazdırılacak ağaç düğümü; indent — girinti düzeyi
        Çıktı:  yok (konsola yazdırma yan etkisi)
        """
        print("  " * indent + f"└─ {node['class']} (id={node['id']})")
        for child in node.get('children', []):
            print_tree(child, indent + 1)

    for root in graph['hierarchy']['roots'][:3]:  # Show first 3 roots
        print_tree(root)


if __name__ == "__main__":
    main()