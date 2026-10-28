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
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(4.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Button(
                onClick = { /* TODO */ },
                modifier = Modifier.padding(8.dp)
            ) {
                Text("Button")
            }

            Text(
                text = "Text Content",
                modifier = Modifier.padding(4.dp)
            )

        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(4.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(
                modifier = Modifier
                    .weight(1f)
                    .padding(4.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "Text Content",
                    modifier = Modifier.padding(4.dp)
                )

                Text(
                    text = "Text Content",
                    modifier = Modifier.padding(4.dp)
                )

            }

            var checked0 by remember { mutableStateOf(false) }
            Checkbox(
                checked = checked0,
                onCheckedChange = { checked0 = it }
            )

        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(4.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Text Content",
                modifier = Modifier.padding(4.dp)
            )

            var checked1 by remember { mutableStateOf(false) }
            Checkbox(
                checked = checked1,
                onCheckedChange = { checked1 = it }
            )

        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(4.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Text Content",
                modifier = Modifier.padding(4.dp)
            )

            var checked2 by remember { mutableStateOf(false) }
            Checkbox(
                checked = checked2,
                onCheckedChange = { checked2 = it }
            )

        }

        Image(
            painter = ColorPainter(Color.LightGray),
            contentDescription = "Image placeholder",
            modifier = Modifier.size(100.dp)
        )

        Text(
            text = "Text Content",
            modifier = Modifier.padding(4.dp)
        )

    }

}