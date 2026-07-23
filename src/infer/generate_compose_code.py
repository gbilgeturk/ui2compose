import json
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ComposeComponent:
    """Jetpack Compose component representation"""
    id: int
    class_name: str
    bbox: List[float]
    children: List['ComposeComponent']
    attributes: Dict = None


class ComposeCodeGenerator:
    """Generates Jetpack Compose code from UI graph"""

    # Component mapping
    COMPONENT_MAP = {
        'Button': 'Button',
        'TextView': 'Text',
        'EditText': 'TextField',
        'ImageView': 'Image',
        'CheckBox': 'Checkbox',
        'Switch': 'Switch',
        'RadioButton': 'RadioButton',
        'Toolbar': 'TopAppBar',
        'Container': 'Box',
        'LinearLayout': 'Column',  # or Row depending on orientation
        'RelativeLayout': 'Box',
        'ConstraintLayout': 'Box',
        'ScrollView': 'Column',  # with verticalScroll modifier
        'CardView': 'Card',
        'SyntheticRow': 'Row',
        'SyntheticColumn': 'Column',
    }

    def __init__(self, indent_size: int = 4):
        """Kod üreticisini girinti genişliğiyle başlatır.

        Girdi:  indent_size — bir girinti düzeyindeki boşluk sayısı
        Çıktı:  yok (girinti dizesi ve kod satırı listesi hazırlanır yan etkisi)
        """
        self.indent = " " * indent_size
        self.code_lines = []

    def generate(self, graph: Dict, use_xy_cut: bool = True) -> str:
        """UI grafından eksiksiz Jetpack Compose ekran kodunu üretir.

        Girdi:  graph — build_ui_graph.py'den gelen UI grafı;
                use_xy_cut — True ise (varsayılan) tespit edilen bileşen kutularından
                özyinelemeli XY-cut (izdüşüm) ile iç içe Row/Column/Box yerleşimi
                kurulur; False ise kökler düz sırayla yazılır
        Çıktı:  tam Kotlin kodu (tek string)
        """
        self.code_lines = []
        self._state_counter = 0  # ensures unique local var names for stateful widgets

        # Header
        self._add_imports()
        self._add_blank_line()

        # Main composable function
        self._add_line("@Composable")
        self._add_line("fun GeneratedScreen() {")

        # Process roots
        hierarchy = graph['hierarchy']
        if use_xy_cut:
            leaves = self._collect_leaves(hierarchy['roots'])
            if leaves:
                tree = self._xy_cut(leaves, prefer='y')
                self._generate_tree(tree, level=1, in_row=False)
            else:
                for root_node in hierarchy['roots']:
                    self._generate_component(root_node, level=1)
        else:
            for root_node in hierarchy['roots']:
                self._generate_component(root_node, level=1)

        self._add_line("}")

        return "\n".join(self.code_lines)

    # ------------------------------------------------------------------
    # Layout synthesis (XY-cut): recover nested Row/Column grouping from
    # the spatial relationships encoded in the component bounding boxes.
    # This operationalizes the rule "vertical separation -> Column,
    # horizontal separation -> Row" described in Section 3.6, so that
    # sibling components (not only contained children) are grouped.
    # ------------------------------------------------------------------
    def _collect_leaves(self, roots: List[Dict]) -> List[Dict]:
        """Hiyerarşiyi düzleştirip atomik (yaprak) bileşen kutularını toplar; XY-cut'ın girdisini hazırlar.

        Girdi:  roots — hiyerarşinin kök düğümleri listesi
        Çıktı:  id, class ve bbox alanlı yaprak sözlükleri listesi
        """
        leaves: List[Dict] = []

        def walk(node: Dict):
            """Alt ağacı gezerek çocuğu olmayan düğümleri leaves listesine ekler.

            Girdi:  node — gezilecek hiyerarşi düğümü
            Çıktı:  yok (leaves listesine ekleme yan etkisi)
            """
            children = node.get('children', [])
            if children:
                for c in children:
                    walk(c)
            else:
                leaves.append({
                    'id': node['id'],
                    'class': node['class'],
                    'bbox': node['bbox'],  # [x1, y1, x2, y2]
                })

        for r in roots:
            walk(r)
        return leaves

    @staticmethod
    def _split_on_gap(boxes: List[Dict], axis: str) -> Optional[List[List[Dict]]]:
        """Kutular arasında temiz bir boşluk varsa verilen eksende >=2 gruba böler (XY-cut'ın tek kesim adımı).

        axis='y' → yatay kesim (gruplar alt alta → Column)
        axis='x' → dikey kesim (gruplar yan yana → Row)

        Girdi:  boxes — bbox alanlı kutu sözlükleri; axis — 'y' veya 'x'
        Çıktı:  grup listelerinin listesi (>=2 grup) ya da temiz kesim yoksa None
        """
        lo, hi = (1, 3) if axis == 'y' else (0, 2)
        ordered = sorted(boxes, key=lambda b: b['bbox'][lo])
        groups: List[List[Dict]] = []
        cur: List[Dict] = [ordered[0]]
        running_max = ordered[0]['bbox'][hi]
        for b in ordered[1:]:
            if b['bbox'][lo] > running_max:        # nothing straddles the gap
                groups.append(cur)
                cur = [b]
                running_max = b['bbox'][hi]
            else:
                cur.append(b)
                running_max = max(running_max, b['bbox'][hi])
        groups.append(cur)
        return groups if len(groups) >= 2 else None

    def _xy_cut(self, boxes: List[Dict], prefer: str = 'y') -> Dict:
        """Bileşen kutularından özyinelemeli XY-cut ile Row/Column yerleşim ağacı kurar.

        Girdi:  boxes — yaprak kutu sözlükleri; prefer — önce denenecek kesim ekseni ('y' veya 'x')
        Çıktı:  {'type': 'leaf'/'Row'/'Column', ...} biçiminde yerleşim ağacı sözlüğü
        """
        if len(boxes) == 1:
            return {'type': 'leaf', 'node': boxes[0]}

        order = ['y', 'x'] if prefer == 'y' else ['x', 'y']
        for axis in order:
            groups = self._split_on_gap(boxes, axis)
            if groups:
                container = 'Column' if axis == 'y' else 'Row'
                nxt = 'x' if axis == 'y' else 'y'   # alternate axis going down
                return {
                    'type': container,
                    'children': [self._xy_cut(g, prefer=nxt) for g in groups],
                }
        # Overlapping boxes: no clean cut -> honest vertical ordering
        ordered = sorted(boxes, key=lambda b: (b['bbox'][1] + b['bbox'][3]) / 2)
        return {'type': 'Column',
                'children': [{'type': 'leaf', 'node': b} for b in ordered]}

    def _generate_tree(self, tree: Dict, level: int, in_row: bool = False):
        """XY-cut ile kurulan yerleşim ağacını Kotlin'e döker; yapraklarda leaf bileşen üretimini yeniden kullanır.

        Girdi:  tree — _xy_cut çıktısı yerleşim ağacı; level — girinti düzeyi;
                in_row — düğüm bir Row çocuğuysa True (weight(1f) uygulanır)
        Çıktı:  yok (code_lines listesine Kotlin satırları eklenir yan etkisi)
        """
        if tree['type'] == 'leaf':
            node = tree['node']
            compose_name = self.COMPONENT_MAP.get(node['class'], 'Box')
            self._generate_leaf_component(node, compose_name, level)
            return

        indent = self.indent * level
        ctype = tree['type']
        self._add_line(f"{indent}{ctype}(")
        self._add_line(f"{indent}    modifier = Modifier")
        # A child of a Row shares horizontal space via weight; otherwise it
        # spans the available width.
        if in_row:
            self._add_line(f"{indent}        .weight(1f)")
        else:
            self._add_line(f"{indent}        .fillMaxWidth()")
        self._add_line(f"{indent}        .padding(4.dp),")
        if ctype == 'Row':
            self._add_line(f"{indent}    horizontalArrangement = Arrangement.spacedBy(8.dp),")
            self._add_line(f"{indent}    verticalAlignment = Alignment.CenterVertically")
        else:
            self._add_line(f"{indent}    verticalArrangement = Arrangement.spacedBy(8.dp),")
            self._add_line(f"{indent}    horizontalAlignment = Alignment.CenterHorizontally")
        self._add_line(f"{indent}) {{")
        for child in tree['children']:
            self._generate_tree(child, level + 1, in_row=(ctype == 'Row'))
        self._add_line(f"{indent}}}")
        self._add_blank_line()

    def _add_imports(self):
        """Gerekli Compose import satırlarını kod listesine ekler.

        Girdi:  yok
        Çıktı:  yok (code_lines listesine import satırları eklenir yan etkisi)
        """
        imports = [
            "import androidx.compose.foundation.Image",
            "import androidx.compose.foundation.layout.*",
            "import androidx.compose.material3.*",
            "import androidx.compose.ui.graphics.Color",
            "import androidx.compose.ui.graphics.painter.ColorPainter",
            "import androidx.compose.runtime.*",
            "import androidx.compose.ui.Modifier",
            "import androidx.compose.ui.unit.dp",
            "import androidx.compose.ui.Alignment",
        ]
        for imp in imports:
            self._add_line(imp)

    def _is_horizontal_layout(self, children: List[Dict]) -> bool:
        """Çocukların bbox merkez dağılımına bakarak yatay mı dizildiklerini belirler (Row/Column kararı).

        Girdi:  children — bbox alanlı çocuk düğümleri listesi
        Çıktı:  True/False — x yayılımı y yayılımından büyükse True (yatay → Row)
        """
        if len(children) < 2:
            return False
        bboxes = [c.get('bbox', [0, 0, 0, 0]) for c in children]
        # Check if children centers are roughly on the same y-level
        cy_values = [(b[1] + b[3]) / 2 for b in bboxes]
        cx_values = [(b[0] + b[2]) / 2 for b in bboxes]
        y_spread = max(cy_values) - min(cy_values) if cy_values else 0
        x_spread = max(cx_values) - min(cx_values) if cx_values else 0
        return x_spread > y_spread

    def _generate_component(self, node: Dict, level: int = 0):
        """Tek bir bileşen ve çocukları için Compose kodunu üretir (kapsayıcı/yaprak ayrımını yaparak).

        Girdi:  node — hiyerarşi düğümü; level — girinti düzeyi
        Çıktı:  yok (code_lines listesine Kotlin satırları eklenir yan etkisi)
        """
        class_name = node['class']
        component_id = node['id']
        children = node.get('children', [])

        # Get Compose equivalent
        compose_name = self.COMPONENT_MAP.get(class_name, 'Box')

        # Handle containers with children
        if children:
            if class_name == 'CardView':
                self._generate_card(node, children, level)
            elif class_name in ['LinearLayout', 'Container', 'ScrollView',
                                'SyntheticRow', 'SyntheticColumn']:
                self._generate_container(node, compose_name, children, level)
            else:
                self._generate_container(node, 'Box', children, level)
        else:
            # Leaf components
            self._generate_leaf_component(node, compose_name, level)

    def _generate_card(self, node: Dict, children: List[Dict], level: int):
        """CardView için Card composable'ını ve içindeki Row/Column yerleşimini üretir.

        Girdi:  node — CardView düğümü; children — çocuk düğümler; level — girinti düzeyi
        Çıktı:  yok (code_lines listesine Kotlin satırları eklenir yan etkisi)
        """
        indent = self.indent * level
        is_horizontal = self._is_horizontal_layout(children)
        inner = 'Row' if is_horizontal else 'Column'

        self._add_line(f"{indent}Card(")
        self._add_line(f"{indent}    modifier = Modifier")
        self._add_line(f"{indent}        .fillMaxWidth()")
        self._add_line(f"{indent}        .padding(8.dp),")
        self._add_line(f"{indent}    elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)")
        self._add_line(f"{indent}) {{")

        # Inner layout
        self._add_line(f"{indent}    {inner}(")
        self._add_line(f"{indent}        modifier = Modifier")
        self._add_line(f"{indent}            .fillMaxWidth()")
        self._add_line(f"{indent}            .padding(12.dp),")
        if is_horizontal:
            self._add_line(f"{indent}        verticalAlignment = Alignment.CenterVertically")
        else:
            self._add_line(f"{indent}        horizontalAlignment = Alignment.CenterHorizontally")
        self._add_line(f"{indent}    ) {{")

        for child in children:
            self._generate_component(child, level + 2)

        self._add_line(f"{indent}    }}")
        self._add_line(f"{indent}}}")
        self._add_blank_line()

    def _generate_container(self, node: Dict, compose_name: str, children: List[Dict], level: int):
        """Kapsayıcı bileşenin (Column, Row, Box) kodunu üretip çocuklarını özyinelemeli işler.

        Girdi:  node — kapsayıcı düğüm; compose_name — eşlenen Compose adı;
                children — çocuk düğümler; level — girinti düzeyi
        Çıktı:  yok (code_lines listesine Kotlin satırları eklenir yan etkisi)
        """
        indent = self.indent * level

        # Determine if it's vertical or horizontal
        if compose_name == 'Row':
            container_type = 'Row'
        elif compose_name == 'Column':
            container_type = 'Column'
        elif self._is_horizontal_layout(children):
            container_type = 'Row'
        else:
            container_type = 'Column'

        # Start container
        self._add_line(f"{indent}{container_type}(")
        self._add_line(f"{indent}    modifier = Modifier")
        self._add_line(f"{indent}        .fillMaxWidth()")

        # Add padding based on layout type
        if node['class'] == 'Toolbar':
            self._add_line(f"{indent}        .padding(16.dp)")
        else:
            self._add_line(f"{indent}        .padding(8.dp)")

        # Alignment
        if container_type == 'Column':
            self._add_line(f"{indent}        ,")
            self._add_line(f"{indent}    horizontalAlignment = Alignment.CenterHorizontally")
        elif container_type == 'Row':
            self._add_line(f"{indent}        ,")
            self._add_line(f"{indent}    verticalAlignment = Alignment.CenterVertically")

        self._add_line(f"{indent}) {{")

        # Generate children
        for child in children:
            self._generate_component(child, level + 1)

        self._add_line(f"{indent}}}")
        self._add_blank_line()

    def _generate_leaf_component(self, node: Dict, compose_name: str, level: int):
        """Yaprak bileşen (Button, Text, TextField vb.) için placeholder içerikli Compose kodunu üretir.

        Girdi:  node — yaprak düğüm; compose_name — eşlenen Compose adı; level — girinti düzeyi
        Çıktı:  yok (code_lines listesine Kotlin satırları eklenir yan etkisi)
        """
        indent = self.indent * level
        class_name = node['class']

        if compose_name == 'Button':
            self._add_line(f'{indent}Button(')
            self._add_line(f'{indent}    onClick = {{ /* TODO */ }},')
            self._add_line(f'{indent}    modifier = Modifier.padding(8.dp)')
            self._add_line(f'{indent}) {{')
            self._add_line(f'{indent}    Text("Button")')
            self._add_line(f'{indent}}}')

        elif compose_name == 'Text':
            self._add_line(f'{indent}Text(')
            self._add_line(f'{indent}    text = "Text Content",')
            self._add_line(f'{indent}    modifier = Modifier.padding(4.dp)')
            self._add_line(f'{indent})')

        elif compose_name == 'TextField':
            n = self._state_counter
            self._state_counter += 1
            var = f'textValue{n}'
            self._add_line(f'{indent}var {var} by remember {{ mutableStateOf("") }}')
            self._add_line(f'{indent}TextField(')
            self._add_line(f'{indent}    value = {var},')
            self._add_line(f'{indent}    onValueChange = {{ {var} = it }},')
            self._add_line(f'{indent}    label = {{ Text("Input") }},')
            self._add_line(f'{indent}    modifier = Modifier.fillMaxWidth().padding(8.dp)')
            self._add_line(f'{indent})')

        elif compose_name == 'Image':
            self._add_line(f'{indent}Image(')
            self._add_line(f'{indent}    painter = ColorPainter(Color.LightGray),')
            self._add_line(f'{indent}    contentDescription = "Image placeholder",')
            self._add_line(f'{indent}    modifier = Modifier.size(100.dp)')
            self._add_line(f'{indent})')

        elif compose_name == 'Checkbox':
            n = self._state_counter
            self._state_counter += 1
            var = f'checked{n}'
            self._add_line(f'{indent}var {var} by remember {{ mutableStateOf(false) }}')
            self._add_line(f'{indent}Checkbox(')
            self._add_line(f'{indent}    checked = {var},')
            self._add_line(f'{indent}    onCheckedChange = {{ {var} = it }}')
            self._add_line(f'{indent})')

        elif compose_name == 'Switch':
            n = self._state_counter
            self._state_counter += 1
            var = f'switched{n}'
            self._add_line(f'{indent}var {var} by remember {{ mutableStateOf(false) }}')
            self._add_line(f'{indent}Switch(')
            self._add_line(f'{indent}    checked = {var},')
            self._add_line(f'{indent}    onCheckedChange = {{ {var} = it }}')
            self._add_line(f'{indent})')

        elif compose_name == 'TopAppBar':
            self._add_line(f'{indent}TopAppBar(')
            self._add_line(f'{indent}    title = {{ Text("App Title") }}')
            self._add_line(f'{indent})')

        else:
            # Generic box for unknown types
            self._add_line(f'{indent}Box(')
            self._add_line(f'{indent}    modifier = Modifier.padding(8.dp)')
            self._add_line(f'{indent}) {{')
            self._add_line(f'{indent}    Text("Unknown: {class_name}")')
            self._add_line(f'{indent}}}')

        self._add_blank_line()

    def _add_line(self, line: str):
        """Kod listesine tek satır ekler.

        Girdi:  line — eklenecek Kotlin satırı
        Çıktı:  yok (code_lines listesine ekleme yan etkisi)
        """
        self.code_lines.append(line)

    def _add_blank_line(self):
        """Kod listesine boş satır ekler.

        Girdi:  yok
        Çıktı:  yok (code_lines listesine boş satır ekleme yan etkisi)
        """
        self.code_lines.append("")

    def save_to_file(self, code: str, output_path: str):
        """Üretilen Compose kodunu UTF-8 olarak dosyaya kaydeder.

        Girdi:  code — Kotlin kodu (string); output_path — hedef .kt dosya yolu
        Çıktı:  yok (diske yazma ve konsol bildirimi yan etkisi)
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"✓ Compose code saved: {output_path}")


def main():
    """Kod üreticisini kayıtlı UI grafı üzerinde uçtan uca dener.

    Girdi:  yok (output/ui_graph.json dosyasından okur)
    Çıktı:  yok (output/GeneratedScreen.kt kaydetme ve konsola yazdırma yan etkisi)
    """
    # Load graph from previous step
    with open("output/ui_graph.json", 'r') as f:
        graph = json.load(f)

    # Generate code
    generator = ComposeCodeGenerator()
    code = generator.generate(graph)

    print("\n📝 Generated Jetpack Compose Code:\n")
    print("=" * 60)
    print(code)
    print("=" * 60)

    # Save to file
    generator.save_to_file(code, "output/GeneratedScreen.kt")


if __name__ == "__main__":
    main()