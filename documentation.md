# 1 Decision Process

To find a suitable paper, we browsed the Journal Club forum in GRIPS and assessed which projects could realistically be completed within two weeks using the knowledge and hardware we had available. Ultimately, we chose *The Space Between the Notes: Adding Expressive Pitch Control to the Piano Keyboard* by McPherson et al [[1]](#references). This paper explores the addition of expressive pitch control such as pitch bends and vibratos to piano keyboards. McPherson et al. achieved this by adding capacitive touch sensors to the keys, allowing finger position and movement to modulate pitch. Considering the concepts we had learned in class, we felt confident that we could reconstruct the project using a computer vision-based approach.

# 2 Apparatus

<img src="docs/images/setup_sketch.png" alt="Setup sketch showing camera, piano area and user" width="80%" />

To replicate the setup you need **4** 6x6 ArUco markers, a camera, a computer, and a flat surface.  
The camera should be positioned in front of the user, facing the piano area at a downward angle.   
The ArUco markers should be placed anywhere in the area that the camera sees forming a rectangle.  
As long as the camera can see all markers and is facing them at a downward angle of roughly `45°`, the application will work as intended.  
You can optionally place the markers at the corners of an actual piano, print a piano keyboard on a piece of paper, or simply use the application window as a reference (it draws a piano keyboard).

# 3 Pipeline

<img src="docs/images/pipeline.png" alt="Pipeline sketch" width="80%" />

The pipeline in this sketch is simplified and shows only the core steps required.  
Additional steps like frame layering or history tracking are not included but described below.

## 3.1 ArUco Marker Detection

We use ArUco markers to define the boundaries of the keyboard area. To ensure robust detection even when markers are temporarily obscured or out of view, we implemented indefinite marker caching, preserving the last known positions to maintain application stability. The markers are sorted consistently to ensure that the digital keyboard is always displayed in the correct orientation, regardless of the physical arrangement of the markers.

## 3.2 Hand Landmark Detection

Hand landmarks are detected using the MediaPipe framework, which supports tracking both hands at the same time. For the purpose of identifying key presses, only the fingertip and the finger base coordinates are extracted from the tracking data (see [3.5](#35-key-press-detection)).

## 3.3 Inverse Hand Landmark Perspective Transformation

TODO: describe how and why the hand landmarks are being transformed to the area defined by the markers.

## 3.4 Fingertip to Key Mapping and Digital Piano Keyboard

TODO: explain that the digital keyboard is mathematically defined and stretched to fit the area defined by the markers. Since (explained in 3.3) the hand landmarks are perspective transformed to that area as well we can simply compare the key positions with the fingertip positions to determine which key is pressed.



## 3.5 Key Press Detection

<img src="docs/images/key_press_detection_demo.png" alt="Key press detection demo" width="30%" />

The blue arrows indicate the distance delta that we use to detect if a piano key is pressed or not.  
If a short distance delta followed by a long distance delta is detected, we assume that the key was pressed because the distance gets shorter when the user lifts their finger thanks to the camera angle. This method is not ideal but suffices for a proof of concept; in a real application more camera angles, capacitive sensors or other methods would be used to detect key presses more reliably.

Mediapipe does support 3D hand landmark detection which we tried extensively but it ultimately had too much noise and was not suitable to reliably track clicks.

## 3.6 Media Playback

Sound is generated using the FluidSynth software synthesizer, which plays back instrument samples from a SoundFont 2 file. While pitch bends and vibratos are typically limited to acoustic instruments with continuous pitch (such as wind or string instruments), our approach can simulate expressive pitch control with any instrument represented in a SoundFont 2 file.

## 3.7 Debugging Frame Composition

TODO: explain how the debug frame consists of the original frame overlayed with the keyboard and an additional masked overlay with only the hands to create a 3D effect.

# 4 Design Decisions

TODO
- why camera only
- why not perspective transform before fingertip detection

# References

[1] McPherson, A.P., Gierakowski, A., and Stark, A.M. (2013). The Space Between the Notes: Adding Expressive Pitch Control to the Piano Keyboard. In *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems* (CHI '13). Association for Computing Machinery, New York, NY, USA, 2195–2204. https://doi.org/10.1145/2470654.2481302