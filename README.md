### Real-Time Drowsiness-Detection System

There is still more work to be done as automotive companies strive to reach level 4 autonomy. Given this, driver safety in automobiles is essential and is of prime concern. Numerous instances of drivers being inattentive owing to either shutting their eyes from too much sleep or emotional state are the major causes of accidents. Therefore, why not make a vehicle intelligent by having the infotainment system manage the driver's attention on the road, and enhance the driver's mental and emotional condition both before and while driving, making it more enjoyable and secure.

So, we developed a CNN-based drowsiness detector with Dlib facial feature extractor with 91.6% accuracy. A sub-network based on lightweight architectures for a yawn and blink detection with an EAR threshold of 0.15 was also designed by our team. We ideated a CNN based on attention mechanism to achieve 0.5 deg best-case accuracy across the same FOV on NVGaze 2M images for eye gaze estimation.

### Operation
The software detects drowsiness of the driver, by constantly monitoring Eyes-Aspect-Ratio(EAR), and a voice alert is sent to the Infotainment System if the driver is detected to be drowsy. Following this, the Infotainment System also recommends nearby places for refreshments which includes cafes, restaurants, rest areas, hotels along with the distance. The software also detects the emotions of the driver and the software suggests recommended actions based on that.

### Demo

![drowsiness](https://user-images.githubusercontent.com/56112545/189865699-056990e7-ddd5-4d74-868c-f0278878e419.png)

![Driver_det](https://user-images.githubusercontent.com/56112545/191074318-8fb6ad7a-6718-4545-8818-94129cf897c7.gif)
