// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import TensorFlowLiteTaskVision
import UIKit

class ViewController: UIViewController {
    
    // MARK: Storyboards Connections
    @IBOutlet weak var previewView: PreviewView!
    @IBOutlet weak var overlayView: OverlayView!
    @IBOutlet weak var resumeButton: UIButton!
    @IBOutlet weak var cameraUnavailableLabel: UILabel!
    
    @IBOutlet weak var bottomSheetStateImageView: UIImageView!
    @IBOutlet weak var bottomSheetView: UIView!
    @IBOutlet weak var bottomSheetViewBottomSpace: NSLayoutConstraint!
    
    // MARK: Constants
    private let displayFont = UIFont.systemFont(ofSize: 14.0, weight: .medium)
    private let edgeOffset: CGFloat = 2.0
    private let labelOffset: CGFloat = 10.0
    private let animationDuration = 0.5
    private let collapseTransitionThreshold: CGFloat = -30.0
    private let expandTransitionThreshold: CGFloat = 30.0
    private let colors = [
        UIColor.red,
        UIColor(displayP3Red: 90.0 / 255.0, green: 200.0 / 255.0, blue: 250.0 / 255.0, alpha: 1.0),
        UIColor.green,
        UIColor.orange,
        UIColor.blue,
        UIColor.purple,
        UIColor.magenta,
        UIColor.yellow,
        UIColor.cyan,
        UIColor.brown,
    ]
    
    // MARK: Model config variables
    private var threadCount: Int = 1
    private var detectionModelAttr: ModelTypeAttribute = ConstantsDefault.modelTypeAttr
    private var detectionModelLand: ModelTypeLandmark = ConstantsDefault.modelTypeLandmark
    private var scoreThreshold: Float = ConstantsDefault.scoreThreshold
    private var maxResults: Int = ConstantsDefault.maxResults
    
    // MARK: Instance Variables
    private var initialBottomSpace: CGFloat = 0.0
    
    // Holds the results at any time
    private var result: Result?
    private let inferenceQueue = DispatchQueue(label: "org.tensorflow.lite.inferencequeue")
    private var isInferenceQueueBusy = false
    
    // MARK: Controllers that manage functionality
    private lazy var cameraFeedManager = CameraFeedManager(previewView: previewView)
    private var objectDetectionHelper: ObjectDetectionHelper? = ObjectDetectionHelper(
        modelFileInfoAttr: ConstantsDefault.modelTypeAttr.modelFileInfo,
        modelFileInfoLandmark: ConstantsDefault.modelTypeLandmark.modelFileInfo,
        threadCount: ConstantsDefault.threadCount,
        scoreThreshold: ConstantsDefault.scoreThreshold,
        maxResults: ConstantsDefault.maxResults
    )
    private var inferenceViewController: InferenceViewController?
    
    // MARK: View Handling Methods
    override func viewDidLoad() {
        super.viewDidLoad()
        guard objectDetectionHelper != nil else {
            fatalError("Failed to create the ObjectDetectionHelper. See the console for the error.")
        }
        cameraFeedManager.delegate = self
        overlayView.clearsContextBeforeDrawing = true
        addPanGesture()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        changeBottomViewState()
        cameraFeedManager.checkCameraConfigurationAndStartSession()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        cameraFeedManager.stopSession()
    }
    
    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }
    
    // MARK: Button Actions
    @IBAction func onClickResumeButton(_ sender: Any) {
        cameraFeedManager.resumeInterruptedSession { (complete) in
            
            if complete {
                self.resumeButton.isHidden = true
                self.cameraUnavailableLabel.isHidden = true
            } else {
                self.presentUnableToResumeSessionAlert()
            }
        }
    }
    
    func presentUnableToResumeSessionAlert() {
        let alert = UIAlertController(
            title: "Unable to Resume Session",
            message: "There was an error while attempting to resume session.",
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
        
        self.present(alert, animated: true)
    }
    
    // MARK: Storyboard Segue Handlers
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        super.prepare(for: segue, sender: sender)
        
        if segue.identifier == "EMBED" {
            inferenceViewController = segue.destination as? InferenceViewController
            
            inferenceViewController?.currentThreadCount = threadCount
            inferenceViewController?.maxResults = maxResults
            inferenceViewController?.scoreThreshold = scoreThreshold
            inferenceViewController?.modelSelectIndex =
            ModelTypeAttribute.allCases.firstIndex(where: { $0 == detectionModelAttr }) ?? 0
            inferenceViewController?.delegate = self
            
            guard let tempResult = result else {
                return
            }
            inferenceViewController?.inferenceTime = tempResult.inferenceTime
        }
    }
}

// MARK: InferenceViewControllerDelegate Methods
extension ViewController: InferenceViewControllerDelegate {
    func viewController(
        _ viewController: InferenceViewController,
        didReceiveAction action: InferenceViewController.Action
    ) {
        switch action {
        case .changeThreadCount(let threadCount):
            if self.threadCount == threadCount { return }
            self.threadCount = threadCount
        case .changeMaxResults(let maxResults):
            if self.maxResults == maxResults { return }
            self.maxResults = maxResults
        case .changeModel(let detectionModel):
            if self.detectionModelAttr == detectionModel { return }
            self.detectionModelAttr = detectionModel
        case .changeScoreThreshold(let scoreThreshold):
            if self.scoreThreshold == scoreThreshold { return }
            self.scoreThreshold = scoreThreshold
        }
        inferenceQueue.async {
            self.objectDetectionHelper = ObjectDetectionHelper(
                modelFileInfoAttr: self.detectionModelAttr.modelFileInfo,
                // Only one landmark model
                modelFileInfoLandmark: self.detectionModelLand.modelFileInfo,
                threadCount: self.threadCount,
                scoreThreshold: self.scoreThreshold,
                maxResults: self.maxResults
            )
        }
    }
}

// MARK: CameraFeedManagerDelegate Methods
extension ViewController: CameraFeedManagerDelegate {
    
    func didOutput(pixelBuffer: CVPixelBuffer) {
        // Drop current frame if the previous frame is still being processed.
        guard !self.isInferenceQueueBusy else { return }
        
        inferenceQueue.async {
            self.isInferenceQueueBusy = true
            self.detect(pixelBuffer: pixelBuffer)
            self.isInferenceQueueBusy = false
        }
    }
    
    // MARK: Session Handling Alerts
    func sessionRunTimeErrorOccurred() {
        // Handles session run time error by updating the UI and providing a button if session can be manually resumed.
        self.resumeButton.isHidden = false
    }
    
    func sessionWasInterrupted(canResumeManually resumeManually: Bool) {
        // Updates the UI when session is interrupted.
        if resumeManually {
            self.resumeButton.isHidden = false
        } else {
            self.cameraUnavailableLabel.isHidden = false
        }
    }
    
    func sessionInterruptionEnded() {
        // Updates UI once session interruption has ended.
        if !self.cameraUnavailableLabel.isHidden {
            self.cameraUnavailableLabel.isHidden = true
        }
        
        if !self.resumeButton.isHidden {
            self.resumeButton.isHidden = true
        }
    }
    
    func presentVideoConfigurationErrorAlert() {
        let alertController = UIAlertController(
            title: "Configuration Failed", message: "Configuration of camera has failed.",
            preferredStyle: .alert)
        let okAction = UIAlertAction(title: "OK", style: .cancel, handler: nil)
        alertController.addAction(okAction)
        
        present(alertController, animated: true, completion: nil)
    }
    
    func presentCameraPermissionsDeniedAlert() {
        let alertController = UIAlertController(
            title: "Camera Permissions Denied",
            message:
                "Camera permissions have been denied for this app. You can change this by going to Settings",
            preferredStyle: .alert)
        
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        let settingsAction = UIAlertAction(title: "Settings", style: .default) { (action) in
            
            UIApplication.shared.open(
                URL(string: UIApplication.openSettingsURLString)!, options: [:], completionHandler: nil)
        }
        
        alertController.addAction(cancelAction)
        alertController.addAction(settingsAction)
        
        present(alertController, animated: true, completion: nil)
        
    }
    
    /** This method runs the live camera pixelBuffer through tensorFlow to get the result.
     */
    func detect(pixelBuffer: CVPixelBuffer) {
        result = self.objectDetectionHelper?.detect(frame: pixelBuffer)
        
        guard let displayResult = result else { return }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        DispatchQueue.main.async {
            
            // Display results by handing off to the InferenceViewController
            self.inferenceViewController?.resolution = CGSize(width: width, height: height)
            
            var inferenceTime: Double = 0
            if let resultInferenceTime = self.result?.inferenceTime {
                inferenceTime = resultInferenceTime
            }
            self.inferenceViewController?.inferenceTime = inferenceTime
            self.inferenceViewController?.tableView.reloadData()
            
            // Draws the bounding boxes and displays class names and confidence scores.
            self.drawAfterPerformingCalculations(
                probs: displayResult.probs,
                detections: displayResult.detections,
                withImageSize: CGSize(width: CGFloat(width), height: CGFloat(height)))
        }
    }
    
    /**
     This method takes the results, translates the bounding box rects to the current view, draws the bounding boxes, classNames and confidence scores of inferences.
     */
    func drawAfterPerformingCalculations(
        probs: [[[Float32]]], detections: [Detection], withImageSize: CGSize
    ) {
        
        self.overlayView.objectOverlays = []
        self.overlayView.setNeedsDisplay()
        
        guard  !probs.isEmpty else {
            return
        }
        
        var objectOverlays: [ObjectOverlay] = []
        
        var index = 0 // index of the for loop
        for detection in detections{
            // Bounding box from detection
            guard let category = detection.categories.first else { continue }

            // Translates bounding box rect to current view.
            var convertedRect = detection.boundingBox.applying(
              CGAffineTransform(
                scaleX: self.overlayView.bounds.size.width / withImageSize.width,
                y: self.overlayView.bounds.size.height / withImageSize.height))

            if convertedRect.origin.x < 0 {
              convertedRect.origin.x = self.edgeOffset
            }

            if convertedRect.origin.y < 0 {
              convertedRect.origin.y = self.edgeOffset
            }

            if convertedRect.maxY > self.overlayView.bounds.maxY {
              convertedRect.size.height =
                self.overlayView.bounds.maxY - convertedRect.origin.y - self.edgeOffset
            }

            if convertedRect.maxX > self.overlayView.bounds.maxX {
              convertedRect.size.width =
                self.overlayView.bounds.maxX - convertedRect.origin.x - self.edgeOffset
            }
            
            // Format the cloth detection strings
            var objectDescription = ""
            for i in 0 ..< probs[index].count {
                let curProb = probs[index][i]
                let curLabelFile = self.detectionModelAttr.labelFiles[i]
                guard let labelPath_cat = Bundle.main.path(
                    forResource: curLabelFile.name,
                    ofType: curLabelFile.extension)
                    else {print("Failed to load the label file with name: \(curLabelFile.name)."); return}
                var fileContents = try? String(contentsOfFile: labelPath_cat)
                // TODO: may load earlier to save time
                guard var labels = fileContents?.components(separatedBy: "\n") else {return}
                
                for j in labels.indices {
                    labels[j] = labels[j].components(separatedBy: " ")[0]
                    if (curProb[j] < scoreThreshold) {continue}
                    // print(i, "\(labels_cat[j]): \(probCat[j])")
                    
                    if (objectDescription != ""){objectDescription = objectDescription + "\n"}
                    objectDescription = objectDescription + String(format: "\(labels[j]) (%.2f)",curProb[j])
                    
                }
            }
            index = index + 1
            
            // Detection and attributes/categories above threshold exists
            if (objectDescription == ""){print("no detection over treshold"); return}
            
            let size = objectDescription.size(withAttributes: [.font: self.displayFont])
            
            // TODO: maybe change color later
            let displayColor = colors[0]
            
            let objectOverlay = ObjectOverlay(
                name: objectDescription,
                borderRect: convertedRect,
                nameStringSize: size,
                color: displayColor,
                font: self.displayFont)
            objectOverlays.append(objectOverlay)
        }

        self.draw(objectOverlays: objectOverlays)
    }
    
    /** Calls methods to update overlay view with detected bounding boxes and class names.
     */
    func draw(objectOverlays: [ObjectOverlay]) {
        
        self.overlayView.objectOverlays = objectOverlays
        self.overlayView.setNeedsDisplay()
    }
    
}

// MARK: Bottom Sheet Interaction Methods
extension ViewController {
    
    // MARK: Bottom Sheet Interaction Methods
    /**
     This method adds a pan gesture to make the bottom sheet interactive.
     */
    private func addPanGesture() {
        let panGesture = UIPanGestureRecognizer(
            target: self, action: #selector(ViewController.didPan(panGesture:)))
        bottomSheetView.addGestureRecognizer(panGesture)
    }
    
    /** Change whether bottom sheet should be in expanded or collapsed state.
     */
    private func changeBottomViewState() {
        guard let inferenceVC = inferenceViewController else {
            return
        }
        
        if bottomSheetViewBottomSpace.constant == inferenceVC.collapsedHeight
            - bottomSheetView.bounds.size.height
        {
            bottomSheetViewBottomSpace.constant = 0.0
        } else {
            bottomSheetViewBottomSpace.constant =
            inferenceVC.collapsedHeight - bottomSheetView.bounds.size.height
        }
        setImageBasedOnBottomViewState()
    }
    
    /**
     Set image of the bottom sheet icon based on whether it is expanded or collapsed
     */
    private func setImageBasedOnBottomViewState() {
        if bottomSheetViewBottomSpace.constant == 0.0 {
            bottomSheetStateImageView.image = UIImage(named: "down_icon")
        } else {
            bottomSheetStateImageView.image = UIImage(named: "up_icon")
        }
    }
    
    /**
     This method responds to the user panning on the bottom sheet.
     */
    @objc func didPan(panGesture: UIPanGestureRecognizer) {
        // Opens or closes the bottom sheet based on the user's interaction with the bottom sheet.
        let translation = panGesture.translation(in: view)
        
        switch panGesture.state {
        case .began:
            initialBottomSpace = bottomSheetViewBottomSpace.constant
            translateBottomSheet(withVerticalTranslation: translation.y)
        case .changed:
            translateBottomSheet(withVerticalTranslation: translation.y)
        case .cancelled:
            setBottomSheetLayout(withBottomSpace: initialBottomSpace)
        case .ended:
            translateBottomSheetAtEndOfPan(withVerticalTranslation: translation.y)
            setImageBasedOnBottomViewState()
            initialBottomSpace = 0.0
        default:
            break
        }
    }
    
    /**
     This method sets bottom sheet translation while pan gesture state is continuously changing.
     */
    private func translateBottomSheet(withVerticalTranslation verticalTranslation: CGFloat) {
        let bottomSpace = initialBottomSpace - verticalTranslation
        guard
            (bottomSpace <= 0.0)
                && (bottomSpace >=
                    inferenceViewController!.collapsedHeight - bottomSheetView.bounds.size.height)
        else {
            return
        }
        setBottomSheetLayout(withBottomSpace: bottomSpace)
    }
    
    /**
     This method changes bottom sheet state to either fully expanded or closed at the end of pan.
     */
    private func translateBottomSheetAtEndOfPan(withVerticalTranslation verticalTranslation: CGFloat)
    {
        
        // Changes bottom sheet state to either fully open or closed at the end of pan.
        let bottomSpace = bottomSpaceAtEndOfPan(withVerticalTranslation: verticalTranslation)
        setBottomSheetLayout(withBottomSpace: bottomSpace)
    }
    
    /**
     Return the final state of the bottom sheet view (whether fully collapsed or expanded) that is to be retained.
     */
    private func bottomSpaceAtEndOfPan(withVerticalTranslation verticalTranslation: CGFloat)
    -> CGFloat
    {
        
        // Calculates whether to fully expand or collapse bottom sheet when pan gesture ends.
        var bottomSpace = initialBottomSpace - verticalTranslation
        
        var height: CGFloat = 0.0
        if initialBottomSpace == 0.0 {
            height = bottomSheetView.bounds.size.height
        } else {
            height = inferenceViewController!.collapsedHeight
        }
        
        let currentHeight = bottomSheetView.bounds.size.height + bottomSpace
        
        if currentHeight - height <= collapseTransitionThreshold {
            bottomSpace = inferenceViewController!.collapsedHeight - bottomSheetView.bounds.size.height
        } else if currentHeight - height >= expandTransitionThreshold {
            bottomSpace = 0.0
        } else {
            bottomSpace = initialBottomSpace
        }
        
        return bottomSpace
    }
    
    /**
     This method layouts the change of the bottom space of bottom sheet with respect to the view managed by this controller.
     */
    func setBottomSheetLayout(withBottomSpace bottomSpace: CGFloat) {
        view.setNeedsLayout()
        bottomSheetViewBottomSpace.constant = bottomSpace
        view.setNeedsLayout()
    }
    
}

// MARK: - Display handler function

/// TFLite model types for attributes
enum ModelTypeAttribute: CaseIterable {
    case resnet50attr
    case resnet50attrCoarse
    
    var modelFileInfo: FileInfo {
        switch self {
        case .resnet50attr:
            return FileInfo("resnet50-attr", "tflite")
        case .resnet50attrCoarse:
            return FileInfo("tf_resnet50_attr_coarse", "tflite")
        }
    }
    
    var title: String {
        switch self {
        case .resnet50attr:
            return "resnet50attr"
        case .resnet50attrCoarse:
            return "resnet50attrCoarse"
        }
    }
    
    var labelFiles: [FileInfo] {
        switch self {
        case .resnet50attr:
            // output 0 - categories, output 1 - attributes
            return [FileInfo("list_attr_cloth", "txt"),FileInfo("list_category_cloth", "txt")]
        case .resnet50attrCoarse:
            // output 0 - categories, output 1 - attributes
            return [FileInfo("list_category_cloth", "txt"),FileInfo("list_attr_cloth", "txt")]
        }
    }
}


/// TFLite model types for landmarks
enum ModelTypeLandmark: CaseIterable {
    case resnet50landmark
    
    var modelFileInfo: FileInfo {
        switch self {
        case .resnet50landmark:
            return FileInfo("resnet50-landmark", "tflite")
        }
    }
    
    var title: String {
        switch self {
        case .resnet50landmark:
            return "resnet50-landmark"
        }
    }
}

/// Default configuration
struct ConstantsDefault {
    static let modelTypeAttr: ModelTypeAttribute = .resnet50attr
    static let modelTypeLandmark: ModelTypeLandmark = .resnet50landmark
    static let threadCount = 1
    static let scoreThreshold: Float = 0.60
    static let maxResults: Int = 10
    static let theadCountLimit = 10
}
