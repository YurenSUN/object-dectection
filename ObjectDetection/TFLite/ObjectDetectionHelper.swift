// Copyright 2022 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlowLite
import TensorFlowLiteTaskVision


/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct Result {
    let inferenceTime: Double
    let probAttr: [[Float32]]
    let probCat: [[Float32]]
    let detections: [Detection]
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `ObjectDetector`.
class ObjectDetectionHelper: NSObject {
    
    // MARK: Private properties
    /// TensorFlow Lite `ObjectDetector` object for performing object detection using a given model.
    
    private var interpreterAttr: Interpreter
    
    private var interpreterLandmark: Interpreter
    
    private var detector: ObjectDetector
    
    private let colors = [
        UIColor.black,  // 0.0 white
        UIColor.darkGray,  // 0.333 white
        UIColor.lightGray,  // 0.667 white
        UIColor.white,  // 1.0 white
        UIColor.gray,  // 0.5 white
        UIColor.red,  // 1.0, 0.0, 0.0 RGB
        UIColor.green,  // 0.0, 1.0, 0.0 RGB
        UIColor.blue,  // 0.0, 0.0, 1.0 RGB
        UIColor.cyan,  // 0.0, 1.0, 1.0 RGB
        UIColor.yellow,  // 1.0, 1.0, 0.0 RGB
        UIColor.magenta,  // 1.0, 0.0, 1.0 RGB
        UIColor.orange,  // 1.0, 0.5, 0.0 RGB
        UIColor.purple,  // 0.5, 0.0, 0.5 RGB
        UIColor.brown,  // 0.6, 0.4, 0.2 RGB
    ]
    
    // MARK: - Initialization
    
    /// A failable initializer for `ObjectDetectionHelper`.
    ///
    /// - Parameter modelFileInfo: The TFLite model to be used.
    /// - Parameter:
    ///   - threadCount: Number of threads to be used.
    ///   - scoreThreshold: Minimum score of objects to be include in the detection result.
    ///   - maxResults: Maximum number of objects to be include in the detection result.
    /// - Returns: A new instance is created if the model is successfully loaded from the app's main
    /// bundle.
    init?(modelFileInfoAttr: FileInfo, modelFileInfoLandmark: FileInfo, threadCount: Int, scoreThreshold: Float, maxResults: Int) {
        
        // Interpretor for attributes and categories
        var modelFilename = modelFileInfoAttr.name
        // Construct the path to the model file.
        guard let modelPath = Bundle.main.path(
            forResource: modelFileInfoAttr.name,
            ofType: modelFileInfoAttr.extension
        )
        else {
            print("Failed to load the model file with name: \(modelFilename).")
            return nil
        }
        
        do {
            // https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_swift
            // Initialize an interpreter with the model.
            interpreterAttr = try Interpreter(modelPath: modelPath)
            try interpreterAttr.resizeInput(at: 0, to: [1, 3, 224, 224])
            
            // Allocate memory for the model's input `Tensor`s.
            try interpreterAttr.allocateTensors()
        } catch let error {
            print("Failed to create the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        // Interpretor for detections
        modelFilename = modelFileInfoLandmark.name
        // Construct the path to the model file.
        guard let modelPath = Bundle.main.path(
            forResource: modelFileInfoLandmark.name,
            ofType: modelFileInfoLandmark.extension
        )
        else {
            print("Failed to load the model file with name: \(modelFilename).")
            return nil
        }
        
        do {
            // https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_swift
            // Initialize an interpreter with the model.
            interpreterLandmark = try Interpreter(modelPath: modelPath)
            try interpreterLandmark.resizeInput(at: 0, to: [1, 3, 224, 224])
            
            // Allocate memory for the model's input `Tensor`s.
            try interpreterLandmark.allocateTensors()
        } catch let error {
            print("Failed to create the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        // Detector
        // https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v1/1/default/1
        guard let modelPath = Bundle.main.path(
            forResource: "ssd_mobilenet_v1_1_default_1",
            ofType: "tflite"
        ) else {
            print("Failed to load the model file with name: ssd_mobilenet_v1_1_default_1.")
            return nil
        }
        
        // Specify the options for the `Detector`.
        let options = ObjectDetectorOptions(modelPath: modelPath)
        options.classificationOptions.scoreThreshold = scoreThreshold
        options.classificationOptions.maxResults = maxResults
        options.baseOptions.computeSettings.cpuSettings.numThreads = Int(threadCount)
        do {
            // Create the `Detector`.
            detector = try ObjectDetector.detector(options: options)
        } catch let error {
            print("Failed to create the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        super.init()
    }
    
    /// Detect objects from the given frame.
    ///
    /// This method handles all data preprocessing and makes calls to run inference on a given frame
    /// through the `Detector`. It then formats the inferences obtained and returns results
    /// for a successful inference.
    ///
    /// - Parameter pixelBuffer: The target frame.
    /// - Returns: The detected objects and other metadata of the inference.
    func detect(frame pixelBuffer: CVPixelBuffer) -> Result? {
        // Detector - image
        guard let mlImage = MLImage(pixelBuffer: pixelBuffer) else { return nil }
        
        // attributes and categories - image
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let width = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let height = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

        // Arrays to return
        var person_detections: [Detection] = []
        var probAttrs: [[Float32]] = []
        var probCats: [[Float32]] = []
        
        // Detector
        do {
            let detectionResult = try detector.detect(mlImage: mlImage)
            let detections = detectionResult.detections
            for detection in detections{
                // Only proceed with detected person and find attributes/categories of the person
                guard let category = detection.categories.first else { continue }
                print(category.label)
                if (category.label != "person") {continue}
                // Append current detection
                person_detections.append(detection)
                
                // Proceed to predict attributes and categories
                let convertedRect = detection.boundingBox
                let origin = convertedRect.origin
                let size = convertedRect.size
//                let startX = Int(origin.x * 224 / width)
//                let startY = Int(origin.y * 224 / height)
//                let endX = startX + Int(size.width * 224 / width)
//                let endY = startY + Int(size.height * 224 / height)
                print("origin", origin.x / width, origin.y / height)
                print("size", size.width / width, size.height / height)
                
                // Normalize and transform image, then put into Data object
                // Modified from - https://firebase.google.com/docs/ml/ios/use-custom-models
                let croppedImage = ciImage.cropped(to: convertedRect)
                let resizedCIImage = croppedImage.transformed(by: CGAffineTransform(scaleX: 224 / size.width, y: 224 / size.height))
                let context = CIContext()
                guard let image = context.createCGImage(resizedCIImage, from: resizedCIImage.extent) else {return nil}
                
                guard let context = CGContext(
                    data: nil,
                    width: image.width, height: image.height,
                    bitsPerComponent: 8, bytesPerRow: image.width * 4,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                ) else {
                    return nil
                }
                context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
                guard let imageData = context.data else { print("no CG image"); return nil }
                
                var inputData = Data()
                
                // Image cropped from detected bounding boxes
                for offset_inc in [1,2,3]{
                    for row in 0 ..< 224 {
                        for col in 0 ..< 224 {
                            let offset = 4 * (row * context.width + col)
                            // (Ignore offset 0, the unused alpha channel)
                            let color = imageData.load(fromByteOffset: offset+offset_inc, as: UInt8.self)
                            
                            var normalized = Float32(color) / 255.0
                            
                            let elementSize = MemoryLayout.size(ofValue: normalized)
                            var bytes = [UInt8](repeating: 0, count: elementSize)
                            
                            memcpy(&bytes, &normalized, elementSize)
                            inputData.append(bytes, count: elementSize)
                        }
                    }
                }
                
                // Attribute prediction
                try interpreterAttr.copy(inputData, toInputAt: 0)
                let startDate = Date()
                
                try interpreterAttr.invoke()
                
                // Output probabilities for categories
                var outputTensor = try interpreterAttr.output(at: 0)
                // Copy output to `Data` to process the inference results.
                let outputSize_cat = outputTensor.shape.dimensions.reduce(1, {x, y in x * y})
                let outputData_cat = UnsafeMutableBufferPointer<Float32>.allocate(capacity: outputSize_cat)
                outputTensor.data.copyBytes(to: outputData_cat)
                
                probCats.append(Array(outputData_cat))
                
                // print("Output data cat", Array(outputData_cat), Array(outputData_cat).count)
                
                // Output probabilities for attributes
                outputTensor = try interpreterAttr.output(at: 1)
                let outputSize_attr = outputTensor.shape.dimensions.reduce(1, {x, y in x * y})
                let outputData_attr = UnsafeMutableBufferPointer<Float32>.allocate(capacity: outputSize_attr)
                outputTensor.data.copyBytes(to: outputData_attr)
                
                probAttrs.append(Array(outputData_attr))
                
                // print("Output data 2", Array(outputData_attr), Array(outputData_attr).count)
                
                let interval = Date().timeIntervalSince(startDate) * 1000
                
                return Result(inferenceTime: interval, probAttr: probAttrs, probCat: probCats, detections: person_detections)
            }
        } catch let error {
            print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        
//        // Run inference
//        do {
//            // Attribute prediction
//            try interpreterAttr.copy(inputData, toInputAt: 0)
//            let startDate = Date()
//
//            try interpreterAttr.invoke()
//
//            // Output probabilities for categories
//            var outputTensor = try interpreterAttr.output(at: 0)
//            // Copy output to `Data` to process the inference results.
//            let outputSize_cat = outputTensor.shape.dimensions.reduce(1, {x, y in x * y})
//            let outputData_cat = UnsafeMutableBufferPointer<Float32>.allocate(capacity: outputSize_cat)
//            outputTensor.data.copyBytes(to: outputData_cat)
//
//            // print("Output data cat", Array(outputData_cat), Array(outputData_cat).count)
//
//            // Output probabilities for attributes
//            outputTensor = try interpreterAttr.output(at: 1)
//            let outputSize_attr = outputTensor.shape.dimensions.reduce(1, {x, y in x * y})
//            let outputData_attr = UnsafeMutableBufferPointer<Float32>.allocate(capacity: outputSize_attr)
//            outputTensor.data.copyBytes(to: outputData_attr)
//
//            // print("Output data 2", Array(outputData_attr), Array(outputData_attr).count)
//
//            let interval = Date().timeIntervalSince(startDate) * 1000
//
//            // Attribute prediction
//            try interpreterLandmark.copy(inputData, toInputAt: 0)
//            try interpreterLandmark.invoke()
//
//            // Output probabilities for categories, landmarks at output 1
//            outputTensor = try interpreterLandmark.output(at: 1)
//            // Copy output to `Data` to process the inference results.
//            let outputSize_landmark = outputTensor.shape.dimensions.reduce(1, {x, y in x * y})
//            let outputData_landmark = UnsafeMutableBufferPointer<Float32>.allocate(capacity: outputSize_landmark)
//            outputTensor.data.copyBytes(to: outputData_landmark)
//
//            // Resize to the propotional of image
//            var outputData_landmark_modified = Array(outputData_landmark)
//            for i in 0 ..< Int(outputSize_landmark / 2) {
//                outputData_landmark_modified[i * 2] = outputData_landmark_modified[i * 2] / 224
//                outputData_landmark_modified[1 + 2 * i] = outputData_landmark_modified[1 + 2 * i] / 224
//            }
//
//            // print("output landmark", Array(outputData_landmark), outputData_landmark_modified)
//            return Result(inferenceTime: interval, probAttr: Array(outputData_attr), probCat: Array(outputData_cat), landmark: outputData_landmark_modified)
//
//        } catch let error {
//            print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
//            return nil
//        }
    return nil
    }
}

