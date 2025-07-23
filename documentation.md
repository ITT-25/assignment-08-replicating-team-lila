# 1 Decision Process

# 1.1 Paper Selection

To find a suitable paper, we browsed the Journal Club forum in GRIPS and assessed which projects could realistically be completed within two weeks using the knowledge and hardware we had available. Ultimately, we chose *The Space Between the Notes: Adding Expressive Pitch Control to the Piano Keyboard* by McPherson et al [[1]](#references). This paper explores the addition of expressive pitch control such as pitch bends and vibratos to piano keyboards. McPherson et al. achieved this by adding capacitive touch sensors to the keys, allowing finger position and movement to modulate pitch.

## 1.2 CV-Based Approach instead of Capacitive Sensors

We wanted to have a minimal setup that can be used with minimal hardware requirements and a simple setup.  
Webcams and printed aruco markers are widely available and easy to set up, making them a good choice for our idea.

We purposely chose to slightly alter the approach from the paper by using a CV-based method instead of capacitive sensors. This makes the application more accessible and combines several concepts learned during the ITT course.

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

We use ArUco markers to define the boundaries of the keyboard area. To ensure robust detection even when markers are temporarily obscured or out of view, we implemented indefinite marker caching, preserving the last known positions to maintain application stability.

## 3.2 Hand Landmark Detection

Hand landmarks are detected using the MediaPipe framework, which supports tracking both hands at the same time. For the purpose of identifying key presses, only the fingertip and the finger base coordinates are extracted from the tracking data (see [3.5](#35-key-press-detection)).

From the detected hand landmarks, we extract only the landmark data for the base and tip of each finger `(4, 1),(8, 5), (12, 9),(16, 13), (20, 17)`.

<img src="https://chuoling.github.io/mediapipe/images/mobile/hand_landmarks.png" alt="Pipeline sketch" width="40%" />

## 3.3 Inverse Hand Landmark Perspective Transformation

TODO: describe how and why the hand landmarks are being transformed to the area defined by the markers.

## 3.4 Fingertip to Key Mapping and Digital Piano Keyboard

Using the area defined by the ArUco markers, the digital keyboard automatically builds itself with the given number of octaves. The position of the keyboard continuously updates itself as well as its size depending on the ArUco markers, making it possible to put the markers closer or farther away in order to scale the keyboard accordingly. Since the hand landmarks are perspective transformed to the keyboard area (see [3.3](#33-inverse-hand-landmark-perspective-transformation)), simply reading out the fingertip coordinates and comparing them to the piano key coordinates allows us to determine whether a fingertip is within the boundaries of a key.

## 3.5 Key Press Detection

<img src="docs/images/key_press_detection_demo.png" alt="Key press detection demo" width="30%" />

The blue arrows indicate the distance delta that we use to detect if a piano key is pressed or not.  
If a short distance delta followed by a long distance delta is detected, we assume that the key was pressed because the distance gets shorter when the user lifts their finger thanks to the camera angle. This method is not ideal but suffices for a proof of concept; in a real application additional camera angles, capacitive sensors or other methods would be used to detect key presses more reliably.

Mediapipe does support 3D hand landmark detection which we tried extensively but it ultimately had too much noise and was not suitable to reliably track clicks.

## 3.6 Media Playback

Sound is generated using the FluidSynth software synthesizer, which plays back instrument samples from a SoundFont 2 file. While pitch bends and vibratos are typically limited to acoustic instruments with continuous pitch (such as wind or string instruments), our approach can simulate expressive pitch control with any instrument represented in a SoundFont 2 file.

Played notes are assigned to their own MIDI channel, which can then be manipulated in terms of pitch. Once the note has finished playing, that MIDI channel is freed for other sounds. This allows for 16 notes to be played and pitched simultaneously.

## 3.7 Pitch Control

Pitch bends are triggered by moving a finger along the y-axis while pressing a key. Sliding the finger upwards increases the pitch, sliding downward decreases it. To avoid accidental triggers, the vertical movement must exceed a defined threshold.

Vibrato is triggered by rapidly moving a finger along the x-axis while keeping it on the key. This causes the pitch to oscillate (or *"wiggle"*). To prevent unintended triggers, several conditions must be met:
- the x-axis movement must exceed a minimum amplitude
- the y-axis movement must stay below a maximum threshold
- the motion must occur within a short time window

## 3.8 Debugging Frame

The dbugging frame that is displayed in the application window, unlike the application logic, does not use the perspective transformation. It shows exactly what the camera sees but uses the inverse projection matrix from the markers to draw the digital piano keyboard as an overlay. To improve the visual quality the hands are masked and overlayed on top of the piano visualization. This allows for a clear view of the piano keys and the hands.

# References

[1] McPherson, A.P., Gierakowski, A., and Stark, A.M. (2013). The Space Between the Notes: Adding Expressive Pitch Control to the Piano Keyboard. In *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems* (CHI '13). Association for Computing Machinery, New York, NY, USA, 2195–2204. https://doi.org/10.1145/2470654.2481302