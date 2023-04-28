// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.example.mysoundclassification

import android.Manifest
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import java.util.*
import kotlin.concurrent.scheduleAtFixedRate

class MainActivity : AppCompatActivity() {
    var TAG = "MainActivity"

    // TODO 2.1: tạo biến có tên mô hình để tải trong các bước tiếp theo.
     var modelPath = "lite-model_yamnet_classification_tflite_1.tflite"

    // TODO 2.2: xác định ngưỡng tối thiểu để chấp nhận thông tin dự đoán từ mô hình
     var probabilityThreshold: Float = 0.3f

    lateinit var textView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val REQUEST_RECORD_AUDIO = 1337
        requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)

        textView = findViewById<TextView>(R.id.output)
        val recorderSpecsTextView = findViewById<TextView>(R.id.textViewAudioRecorderSpecs)

        // TODO 2.3: nơi chúng ta sẽ tải mô hình từ thư mục assets.
         val classifier = AudioClassifier.createFromFile(this, modelPath)

        // TODO 3.1: Tạo biến tensor sẽ lưu trữ bản ghi để dự đoán và xây dựng quy cách định dạng cho trình ghi.
         val tensor = classifier.createInputTensorAudio()

        // TODO 3.2:  Hiển thị thông số kỹ thuật của trình ghi âm được siêu dữ liệu xác định trong bước trước đó của mô hình.
         val format = classifier.requiredTensorAudioFormat
         val recorderSpecs = "Number Of Channels: ${format.channels}\n" +
                "Sample Rate: ${format.sampleRate}"
         recorderSpecsTextView.text = recorderSpecs

        // TODO 3.3: Tạo trình ghi âm và bắt đầu ghi.
         val record = classifier.createAudioRecord()
         record.startRecording()

        Timer().scheduleAtFixedRate(1, 500) {

            // TODO 4.1:  Tải bản ghi âm vào một biến thể âm thanh rồi chuyển bản ghi đó đến bộ phân loại.
             val numberOfSamples = tensor.load(record)
             val output = classifier.classify(tensor)

            // TODO 4.2: Lọc ra các phân loại có xác suất thấp.
             val filteredModelOutput = output[0].categories.filter {
                 it.score > probabilityThreshold
             }

            // TODO 4.3: Tạo một chuỗi nhiều dòng với các kết quả được lọc.
            val outputStr =
                filteredModelOutput.sortedBy { -it.score }
                    .joinToString(separator = "\n") { "${it.label} -> ${it.score} " }

            // TODO 4.4: Cập nhật giao diện người dùng.
            if (outputStr.isNotEmpty())
                runOnUiThread {
                    textView.text = outputStr
                }
        }
    }
}