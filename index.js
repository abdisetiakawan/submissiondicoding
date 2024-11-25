const express = require("express");
const multer = require("multer");
const { Storage } = require("@google-cloud/storage");
const tf = require("@tensorflow/tfjs-node");
const { Firestore } = require("@google-cloud/firestore");
const sharp = require("sharp");
const cors = require("cors");

const app = express();
const port = process.env.PORT;

app.use(cors());

// Konfigurasi Google Cloud Storage
const storage = new Storage();
const bucketName = "mlgcmodel-abdisetiawan";
const modelBucket = storage.bucket(bucketName);

// Konfigurasi Firestore
const firestore = new Firestore();
const predictionsCollection = firestore.collection("predictions");

// Konfigurasi Multer untuk upload
const upload = multer({
  limits: {
    fileSize: 1000000, // 1MB
  },
});

// Fungsi load model dari Cloud Storage
async function loadModelFromGCS() {
  try {
    // Download model dari Cloud Storage
    const [files] = await modelBucket.getFiles({ prefix: "model/" });
    const modelFile = files.find((file) => file.name.endsWith("model.json"));

    if (!modelFile) {
      throw new Error("Model not found in bucket");
    }
    await modelFile.download({ destination: "./model" });
    const model = await tf.loadLayersModel("file://./model/model.json");
    return model;
  } catch (error) {
    console.error("Error loading model:", error);
    throw error;
  }
}

// Fungsi preprocessing gambar
async function preprocessImage(file) {
  try {
    // Resize dan normalisasi gambar
    const imageBuffer = await sharp(file.buffer)
      .resize(224, 224)
      .toFormat("jpeg")
      .toBuffer();

    // Konversi ke tensor
    const tensor = tf.node
      .decodeImage(imageBuffer)
      .toFloat()
      .expandDims(0)
      .div(255.0);

    return tensor;
  } catch (error) {
    console.error("Error preprocessing image:", error);
    throw error;
  }
}

app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        status: "fail",
        message: "Tidak ada gambar yang diunggah",
      });
    }
    const model = await loadModelFromGCS();
    const imageTensor = await preprocessImage(req.file);
    const prediction = model.predict(imageTensor);
    const predictionValue = prediction.dataSync()[0];
    const result = predictionValue > 0.5 ? "Cancer" : "Non-cancer";
    const suggestion =
      result === "Cancer"
        ? "Segera periksa ke dokter!"
        : "Penyakit kanker tidak terdeteksi.";
    const id = generateUniqueId();
    await predictionsCollection.doc(id).set({
      id,
      result,
      suggestion,
      createdAt: new Date().toISOString(),
    });
    res.json({
      status: "success",
      message: "Model is predicted successfully",
      data: {
        id,
        result,
        suggestion,
        createdAt: new Date().toISOString(),
      },
    });
  } catch (error) {
    console.error("Prediction error:", error);
    res.status(400).json({
      status: "fail",
      message: "Terjadi kesalahan dalam melakukan prediksi",
    });
  }
});
function generateUniqueId() {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    var r = (Math.random() * 16) | 0,
      v = c == "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// Start server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
