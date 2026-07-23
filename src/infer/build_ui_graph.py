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
        """Initializes the graph builder with containment and alignment thresholds.

        Input:  iou_threshold — IoU threshold for the containment (parent-child) decision;
                vertical_threshold — pixel threshold for vertical alignment;
                horizontal_threshold — pixel threshold for horizontal alignment
        Output: none (side effect of assigning thresholds to object fields)
        """
        self.iou_threshold = iou_threshold
        self.v_thresh = vertical_threshold
        self.h_thresh = horizontal_threshold

    def build_graph(self, detections: List[Dict]) -> Dict:
        """Builds the full UI graph from detected components: containment, hierarchy, layout groups and spatial relations.

        Input:  detections — list of component dictionaries from YOLO detections
        Output: dictionary {'nodes': components, 'edges': relations, 'hierarchy': parent-child tree}
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
        """Clusters root-level nodes by y/x alignment and groups them in synthetic SyntheticRow/SyntheticColumn containers.

        Input:  hierarchy — hierarchy dictionary containing the 'roots' list
        Output: updated hierarchy dictionary with roots wrapped in synthetic containers
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
        """Merges consecutive rows whose items align vertically (matching x-centers) into vertical groups (_vgroup).

        Example: if ["Reward Points", "Travel Trips", "Bucket List"] (row A) and
               ["360", "238", "473"] (row B) are aligned, they merge into a single row:
               [("Reward Points","360"), ("Travel Trips","238"), ("Bucket List","473")]

        Input:  rows — rows of root nodes (list of lists); avg_height — average height used for x-alignment tolerance
        Output: merged row list (with representative nodes marked with _vgroup)
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
        """Clusters nodes by their x-center positions (comparing against the group average).

        Input:  nodes — node list with _cx field computed; threshold — clustering tolerance (pixels)
        Output: list of node lists grouped along the x axis
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
        """Creates a synthetic container node (SyntheticRow/SyntheticColumn) with a bbox enclosing the children's boxes.

        Input:  node_id — negative synthetic id; class_name — 'SyntheticRow' or 'SyntheticColumn';
                children — child nodes to place in the container
        Output: container node dictionary with id, class, bbox and children fields
        """
        all_bboxes = []
        def collect(n):
            """Collects the bboxes of the node and its whole subtree into the all_bboxes list.

            Input:  n — node dictionary that may have bbox and children fields
            Output: none (side effect of appending to the all_bboxes list)
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
        """Removes the temporary fields (_cy, _cx, _h) used during layout inference from the node.

        Input:  node — node dictionary to clean
        Output: none (side effect of updating the node in place)
        """
        node.pop('_cy', None)
        node.pop('_cx', None)
        node.pop('_h', None)

    def _find_containment_relations(self, nodes: List[Dict]) -> List[Edge]:
        """Finds parent-child edges based on bounding-box containment.

        Input:  nodes — list of component nodes with bbox fields
        Output: list of Edges of type PARENT_CHILD with transitive ones filtered out
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
        """Checks whether the inner box is completely inside the outer box (containment test).

        Input:  inner_box, outer_box — two boxes in (x1, y1, x2, y2) format
        Output: True/False — True if the inner box is inside the outer box
        """
        ix1, iy1, ix2, iy2 = inner_box
        ox1, oy1, ox2, oy2 = outer_box

        # Inner box must be completely inside outer box
        return (ox1 <= ix1 and iy1 >= oy1 and
                ix2 <= ox2 and iy2 <= oy2)

    def _remove_transitive_containment(self, edges: List[Edge], nodes: List[Dict]) -> List[Edge]:
        """Filters out indirect (transitive) parent-child edges, keeping only direct parents.

        Input:  edges — list of PARENT_CHILD edges; nodes — list of component nodes
        Output: list of Edges containing only direct parent-child relations
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
        """Finds root nodes from the containment edges and builds the parent-child tree recursively.

        Input:  nodes — component nodes; containment_edges — PARENT_CHILD edges
        Output: hierarchy dictionary in the form {'roots': [tree nodes]}
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
            """Builds the subtree recursively starting from the given node.

            Input:  node_id — id of the node that will be the root of the subtree
            Output: tree node dictionary with id, class, bbox and children fields
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
        """Finds spatial (above/below/left/right) edges between sibling components sharing the same parent.

        Input:  nodes — component nodes; hierarchy — hierarchy tree from which the parent mapping is extracted
        Output: list of Edges of type ABOVE/BELOW/LEFT_OF/RIGHT_OF
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
        """Determines the spatial relation between two components based on their center positions and alignment thresholds.

        Input:  node1, node2 — two component nodes with 'center' fields
        Output: RelationType (ABOVE/BELOW/LEFT_OF/RIGHT_OF) or None if there is no alignment
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
        """Traverses the hierarchy tree and extracts the child→parent id mapping.

        Input:  hierarchy — hierarchy dictionary containing the 'roots' list
        Output: mapping dictionary in the form {child_id: parent_id}
        """
        parent_map = {}

        def traverse(node, parent_id=None):
            """Traverses the subtree and records each child with its parent in parent_map.

            Input:  node — node to traverse; parent_id — parent id of the node (None at the root)
            Output: none (side effect of writing to the parent_map dictionary)
            """
            if parent_id is not None:
                parent_map[node['id']] = parent_id
            for child in node.get('children', []):
                traverse(child, node['id'])

        for root in hierarchy['roots']:
            traverse(root)

        return parent_map

    def _edge_to_dict(self, edge: Edge) -> Dict:
        """Converts an Edge object into a JSON-serializable dictionary.

        Input:  edge — Edge carrying source, target, relation type and confidence
        Output: dictionary with source, target, relation, confidence keys
        """
        return {
            'source': edge.source_id,
            'target': edge.target_id,
            'relation': edge.relation_type.value,
            'confidence': edge.confidence
        }

    def visualize_graph(self, graph: Dict, output_path: str = "graph.json"):
        """Saves the graph as a JSON file for visualization.

        Input:  graph — dictionary output by build_graph; output_path — target JSON file path
        Output: none (side effect of writing JSON to disk and printing a console notice)
        """
        with open(output_path, 'w') as f:
            json.dump(graph, f, indent=2)
        print(f"✓ Graph saved: {output_path}")


def main():
    """Tries the graph builder end to end on saved detection output and prints the hierarchy.

    Input:  none (reads from the output/detections.json file)
    Output: none (side effect of saving output/ui_graph.json and printing a summary to the console)
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
        """Prints the hierarchy tree to the console in indented form.

        Input:  node — tree node to print; indent — indentation level
        Output: none (side effect of printing to the console)
        """
        print("  " * indent + f"└─ {node['class']} (id={node['id']})")
        for child in node.get('children', []):
            print_tree(child, indent + 1)

    for root in graph['hierarchy']['roots'][:3]:  # Show first 3 roots
        print_tree(root)


if __name__ == "__main__":
    main()