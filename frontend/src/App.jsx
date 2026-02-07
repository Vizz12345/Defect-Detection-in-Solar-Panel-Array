import React, { useState } from "react";
import { Upload, ImageIcon, Loader2 } from "lucide-react";

/* Simple reusable UI components */
const Card = ({ children }) => (
  <div className="bg-white rounded-2xl shadow">{children}</div>
);

const CardContent = ({ children, className = "" }) => (
  <div className={`p-6 ${className}`}>{children}</div>
);

const Button = ({ className = "", ...props }) => (
  <button
    className={`px-4 py-2 rounded-xl bg-blue-600 text-white hover:bg-blue-700 transition ${className}`}
    {...props}
  />
);

export default function SolarDefectDetector() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [imageType, setImageType] = useState("RGB");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  /* 🔥 REAL BACKEND CALL HERE */
  const handlePredict = async () => {
    if (!image) return;
    setIsLoading(true);

    const formData = new FormData();
    formData.append("file", image);
    formData.append("image_type", imageType);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Backend not reachable");
    }

    setIsLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-50">

      {/* NAVBAR */}
      <nav className="w-full bg-white border-b shadow-sm">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <h1 className="text-xl font-bold tracking-wide">
            Solar Panel Defect Detection
          </h1>
        </div>
      </nav>

      {/* MAIN CONTENT */}
      <div className="max-w-6xl mx-auto mt-8 p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

          {/* LEFT PANEL */}
          <Card>
            <CardContent className="flex flex-col gap-4">
              <h2 className="text-xl font-semibold">Upload Panel Image</h2>

              <label className="border-2 border-dashed rounded-xl p-6 cursor-pointer flex flex-col items-center justify-center bg-gray-50 hover:bg-gray-100 transition">
                <Upload className="w-8 h-8 mb-2" />
                <span>Select or Drop Image</span>
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleFileChange}
                />
              </label>

              {/* IMAGE TYPE DROPDOWN */}
              <div>
                <label className="text-sm font-semibold text-gray-600">
                  Image Type
                </label>
                <select
                  value={imageType}
                  onChange={(e) => setImageType(e.target.value)}
                  className="w-full mt-1 p-2 rounded-xl border focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="RGB">RGB Image</option>
                  <option value="EL">EL Image</option>
                  <option value="Thermal">Thermal Image</option>
                </select>
              </div>

              <Button
                className="mt-2 w-fit"
                onClick={handlePredict}
                disabled={!image || isLoading}
              >
                {isLoading ? (
                  <Loader2 className="animate-spin" />
                ) : (
                  "Run Prediction"
                )}
              </Button>
            </CardContent>
          </Card>

          {/* RIGHT PANEL — IMAGE PREVIEW */}
          <Card>
            <CardContent className="p-4 h-full flex items-center justify-center bg-gray-100 rounded-2xl">
              {preview ? (
                <img
                  src={preview}
                  alt="Preview"
                  className="object-cover h-full w-full rounded-2xl"
                />
              ) : (
                <div className="flex flex-col items-center gap-2 text-gray-500">
                  <ImageIcon />
                  <p>No image selected</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* RESULTS */}
        {result && (
          <div className="mt-6 grid grid-cols-1 sm:grid-cols-4 gap-4">
            <div className="bg-white p-4 rounded-xl shadow border-l-4 border-blue-600">
              <p className="text-gray-500 text-sm">Prediction</p>
              <p className="font-bold text-lg">{result.label}</p>
            </div>

            <div className="bg-white p-4 rounded-xl shadow border-l-4 border-blue-600">
              <p className="text-gray-500 text-sm">Confidence</p>
              <p className="font-bold text-lg">{result.confidence}</p>
            </div>

            <div className="bg-white p-4 rounded-xl shadow border-l-4 border-blue-600">
              <p className="text-gray-500 text-sm">Inference Time</p>
              <p className="font-bold text-lg">{result.inference}</p>
            </div>

            <div className="bg-white p-4 rounded-xl shadow border-l-4 border-blue-600">
              <p className="text-gray-500 text-sm">Model Assigned</p>
              <p className="font-bold text-lg">{result.modelUsed}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
