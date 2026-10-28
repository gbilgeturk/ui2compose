import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.painter.ColorPainter
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.Alignment

@Composable
fun GeneratedScreen() {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(4.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Text Content",
            modifier = Modifier.padding(4.dp)
        )

        var textValue0 by remember { mutableStateOf("") }
        TextField(
            value = textValue0,
            onValueChange = { textValue0 = it },
            label = { Text("Input") },
            modifier = Modifier.fillMaxWidth().padding(8.dp)
        )

        var textValue1 by remember { mutableStateOf("") }
        TextField(
            value = textValue1,
            onValueChange = { textValue1 = it },
            label = { Text("Input") },
            modifier = Modifier.fillMaxWidth().padding(8.dp)
        )

        Button(
            onClick = { /* TODO */ },
            modifier = Modifier.padding(8.dp)
        ) {
            Text("Button")
        }

    }

}