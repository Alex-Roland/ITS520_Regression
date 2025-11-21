async function runExample() {

  const x = new Float32Array(2);
  x[0] = parseFloat(document.getElementById('box0c1').value) || 0;
  x[1] = parseFloat(document.getElementById('box1c1').value) || 0;

  const tensorX = new ort.Tensor('float32', x, [1, 2]);

  try {
    const session = await ort.InferenceSession.create("temperature_humidity_data.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data;

    // render here (output is in scope)
    const predictions = document.getElementById('predictions');
    predictions.innerHTML = `
      <table>
        <tr><td>Inside Temp</td>          <td id="c1td0">${output[0].toFixed(2)}</td></tr>
        <tr><td>Inside Humidity</td>      <td id="c1td1">${output[1].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }
}
