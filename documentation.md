# 1 Apparatus

<img src="docs/images/setup_sketch.png" alt="Setup sketch showing camera, piano area and user" width="80%" />

To replicate the setup you need **4** 6x6 Aruco markers, a camera, a computer and a flat surface.  
The camera should be positioned in front of the user, facing the piano area at a downward angle.   
The Aruco markers should be placed anywhere in the area that the camera sees forming a rectangle.  
As long as the camera can see all markers and is looking at them at a downward angle of roughly `45°`, the application will work as intended.  
You can optionally place the markers at the corners of an actual piano, print a piano keyboard on a piece of paper, or simply use the application window as a reference (it draws a piano keyboard).

# 2 Pipeline

<img src="docs/images/pipeline.png" alt="Pipeline sketch" width="80%" />

The pipeline in this sketch is simplified and shows only the core steps required.  
Additional steps like frame layering or history tracking are not included but described below.

## 2.1 Aruco Marker Detection

TODO: quickly explain the we use aruco markers to define the board area, also briefly touch on the measures we took to ensure that markers are detected reliably (indefinite marker caching)

## 2.2 Hand Landmark Detection

TODO: describe how we use mediapipe to detect hand landmarks and that we extract only the fingertip and fingerbase positions for further processing

## 2.3 Inverse Hand Landmark Perspective Transformation

TODO: describe how and why the hand landmarks are being transformed to the area defined by the markers.

## 2.4 Fingertip to Key Mapping and Digital Piano Keyboard

TODO: explain that the digital keyboard is mathematically defined and stretched to fit the area defined by the markers. Since (explained in 2.3) the hand landmarks are perspective transformed to that area as well we can simply compare the key positions with the fingertip positions to determine which key is pressed.

## 2.5 Key Press Detection

<img src="docs/images/key_press_detection_demo.png" alt="Key press detection demo" width="30%" />

The blue arrows indicate the distance delta that we use to detect if a piano key is pressed or not.  
If a short distance delta followed by a long distance delta is detected, we assume that the key was pressed because the distance gets shorter when the user lifts their finger thanks to the camera angle. This method is not ideal but suffices for a proof of concept, in a real application more camera angles, capacitive sensors or other methods would be used to detect key presses more reliably.

Mediapipe does support 3D hand landmark detection which we tried extensively but it ultimately had too much noise and was not suitable to reliably track clicks.

## 2.6 Debugging Frame Composition

TODO: explain how the debug frame consists of the original frame overlayed with the keyboard and an additional masked overlay with only the hands to create a 3D effect.

# 3 Design Decisions

TODO
- why camera only
- why not perspective transform before fingertip detection