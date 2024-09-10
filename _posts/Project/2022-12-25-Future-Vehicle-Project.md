---
title: "[프로젝트] Future Vehicle Project(SLAM & Navigation) Roadmap"
last_modified_at: 2022-12-25
categories:
  - Project
excerpt: "Inha univ, KSAE"
use_math: true
classes: wide
---

<style>
    .boxed {
        border: 2px solid #000;
        padding: 10px;
        display: inline-block;
        margin: 10px;
    }
</style>

> Inha univ.  |  한국자동차공학회(KSAE)  
Paper num :  22AKSAE_D070  
> 

## Review

3차원 지도를 획득하기 위해 스테레오 및 단일 카메라를 활용하여 raw image data에서 특징점을 찾아 포인트 클라우드를 생성하였으며, obstacle이 많은 환경에서도 원활한 지도 획득 성능을 보였다.  

Navigation을 위해 depth filtering, height filtering을 거쳐 3차원 지도를 2차원 지도로 변환하였으며, odometry 및 map 데이터 기반으로 출발지부터 목적지까지 최단 경로를 계획하여 navigation을 수행하였다.  

이번 프로젝트는 ROS의 다양한 기능과 노드 구조를 직접 다루면서 실무적인 경험을 쌓을 수 있는 기회였으며, SLAM 및 네비게이션에 대한 기초적인 개념부터 실제 하드웨어와 소프트웨어를 결합한 응용까지 다양한 측면에서 배우는 계기가 되었고 실제 학회에 포스터를 제출하는 경험을 했다.

---

## Goal

이번 프로젝트는 SLAM & Navigation에 기본 가이드라인을 제공하는 것을 목표로 기본적인 개념과 오류를 정리하여 제공한다.

스테레오 및 단일 카메라를 활용한 Visual SLAM 및 global path planning 방법을 제안한다.

---

## **Platform specs**

- Mobile Robot Platform: TurtleBot3 Burger
- MCU: Nvidia Jetson Nano
    - OS: Ubuntu 18.04 LTS
    - Middleware: ROS melodic
- OpenCR 1.0 (for motor controll …)
- remote PC: NUC
    - OS: Ubuntu 18.04 LTS
    - Middleware: ROS melodic
- Sensor: Intel Realsense T265, Intel Realsense D435i (Camera module)

---

## Seneor

| D400 Stereo Depth Cameras | **T265 : Tracking Camera** |
| --- | --- |
| Intel RealSense D400 시리즈는 **Stereo Depth Cameras**로, 3D 공간에서 물체와의 거리를 측정하는 데 사용된다. 이 카메라는 Stereo Vision 기술을 사용하여 두 개의 카메라로부터 얻은 이미지를 비교해 깊이 데이터를 추출한다. | **Intel RealSense T265**는 **Tracking Camera**로, 주로 **비주얼 SLAM(Simultaneous Localization and Mapping)** 기능을 제공한다. 이는 GPS 없이도 실내외 환경에서 실시간으로 물체의 위치와 방향을 추적하는 기술이다. T265는 특히 로봇, 드론, 자율주행 차량 등의 위치 추적에 강력한 성능을 발휘한다. |  

기본적으로 ROS(로봇 운영 체제)에서 카메라를 사용하려면 **Launcher**를 통해 카메라 노드를 실행하고 필요한 파라미터를 설정한다.

## Sensor test [Intel Realsense T265]

tracking test result

![Untitled](/assets/Images/2022-12-25-Future-Vehicle-Project/Untitled.png){: width="40%"}

2d camera test result

![Untitled](/assets/Images/2022-12-25-Future-Vehicle-Project/Untitled%201.png){: width="40%"}


---

## 1차 task

### “rs_camera.launch”와 “rs_d400_and_t265.launch” 비교 분석

> 각 launch file 및 github repository를 참고하여 두 launch 파일을 비교 분석
launch 실행 시 생성되는 node, topic message 등을 확인하여 자세히 분석할 것
> 
> - rs_camera.launch (from Intel Official Github)
>     
>     [realsense-ros/rs_camera.launch at 2.3.2 · IntelRealSense/realsense-ros](https://github.com/IntelRealSense/realsense-ros/blob/2.3.2/realsense2_camera/launch/rs_camera.launch)
>     
> - rs_d400_and_t265.launch (from Intel Official Github)
>     
>     [realsense-ros/rs_d400_and_t265.launch at 2.3.2 · IntelRealSense/realsense-ros](https://github.com/IntelRealSense/realsense-ros/blob/2.3.2/realsense2_camera/launch/rs_d400_and_t265.launch)
>     

### **rs_camera : base cameras**

Intel RealSense 카메라의 기본 설정과 관련된 개념으로 real sense camera의 모든 parameter 조정이 가능한 런처이다.
RealSense 카메라를 **ROS**(Robot Operating System) 환경에서 사용할 때 나타나는 설정 중 하나로, **카메라의 기본 기능을 활성화**하고 **카메라에서 수집된 데이터를 처리**한다.

카메라 모델 마다 각자의 기능을 수행한다.

입력된 파라미터로 "$(find realsense2_camera)/launch/includes/nodelet.launch.xml" 를 실행

- realsense2_camera node, publishes images of a (device_type) device.

### **rs_T265 and rs_D400**

D400의 공간상의 위치(깊이) 정보와 T265의 트레킹 정보를 동시에 얻어오는 런처

입력된 파라미터로 "$(find realsense2_camera)/launch/includes/nodelet.launch.xml" 를 실행

- realsense2_camera node, publishes images of a D400 device.
- realsense2_camera node, publishes orientation from a T265 device.
- static_transform_publisher node, publishes the known transformation between the D400 and the T265 devices.
- rtabmap node, uses the orientation and images to improve accuracy and 3D-reconstruct the scene.

### **D400+T265 ROS examples**

[https://dev.intelrealsense.com/docs/2d-occupancy-map](https://dev.intelrealsense.com/docs/2d-occupancy-map)

### **D400 : Stereo Depth Cameras luncher**

다음 정보는 intel realsense t400 datasheet 에서 찾아볼 수 있고 parameter, topic을 가진다

[www.intelrealsense.com](https://www.intelrealsense.com/wp-content/uploads/2022/05/Intel-RealSense-D400-Series-Datasheet-April-2022.pdf)

parameter, topic을 가진다

**color , depth/stereo , Extrinsics** , **Left IR Imager / Right IR Imager , PointCloud, IMU, Other**

![T265 – how to get Gyro data](/assets/Images/2022-12-25-Future-Vehicle-Project/image.png)


T265 – how to get Gyro data

### **T265 : Tracking Camera**

[Intel® RealSense™ Tracking Camera T265 Datasheet](https://www.intel.com/content/www/us/en/support/articles/000032422/emerging-technologies/intel-realsense-technology.html)

**Odometry , Tracking, Fisheye , IMU , Other**

---

## 2차 task [mapping, filtering]

### Mapping

실제 Tutlebot을 가지고 Mapping 진행

<div style="display: flex; justify-content: space-between;">
    <img src="/assets/Images/2022-12-25-Future-Vehicle-Project/mapping_rqt.png" style="width: 49%;">
    <img src="/assets/Images/2022-12-25-Future-Vehicle-Project/nav_rqt.png" style="width: 49%;">
</div>

<div style="display: flex; justify-content: space-between;">
    <img src="/assets/Images/2022-12-25-Future-Vehicle-Project/Screenshot_from_2022-11-14_15-55-41.png" style="width: 49%;">
    <img src="/assets/Images/2022-12-25-Future-Vehicle-Project/Screenshot_from_2022-11-14_15-58-17.png" style="width: 49%;">
</div>

<div style="display: flex; justify-content: space-between;">
    <img src="/assets/Images/2022-12-25-Future-Vehicle-Project/Screenshot_from_2022-11-14_16-06-28.png" style="width: 49%;">
</div>



### Filtering

Navigation을 위해서는 다음 Map에서 천장과 바닥면의 point cloud를 제거해야한다.

**height filtering 방법 고안**

1. **pcl filtering**
    
    point cloud python reference
    
    [Python Bindings to the Point Cloud Library](https://strawlab.github.io/python-pcl/)
    
    아래 튜토리얼에서 제안된  3D 데이터 전처리, 필터링 방법 사용
    
    [INTRO](https://pcl.gitbook.io/tutorial/)
    
    ![Untitled](/assets/Images/2022-12-25-Future-Vehicle-Project/Untitled%202.png)
    
    다음 PCL 에 구현되어있는 면 모델을 이용하여 바닥과 벽 천장 감지 가능

    ---
    
2.  **pcl 파일 정렬 순서 이용**
    
    PCL의 WIDTH, POINTS 입력 값 변경을 통해 읽는 point 개수 조절 가능
    
    200000개 point 사용
    
    ![20000.png](/assets/Images/2022-12-25-Future-Vehicle-Project/20000.png){: width="40%"}
    
    400000개 point(위에서 본 데이터)
    
    ![40000up.png](/assets/Images/2022-12-25-Future-Vehicle-Project/40000up.png){: width="40%"}
    
    PCL 파일이 Data를 저장할 때 바닥면부터 저장함을 알 수 있다.
    
    middle 100000 points
    
    ![100000~200000.png](/assets/Images/2022-12-25-Future-Vehicle-Project/100000200000.png){: width="40%"}
    
    - 따라서 모든 입력을 받은 후 WIDTH, POINTS 값 조절을 통해 천장면 삭제 가능
    - 혹은 vi 편집기 명령어를 통해 중간 벽 point만 추출
        
        [https://blockdmask.tistory.com/25](https://blockdmask.tistory.com/25)

    ---

3. **ros_pcl 이용**
    
    PassThrough Pcl_manager 
    
    [Wiki](http://wiki.ros.org/pcl_ros/Tutorials/PassThrough%20filtering)
    
    ---
    
4. **카메라 각도 이용**
    
    ![Untitled](/assets/Images/2022-12-25-Future-Vehicle-Project/Untitled%203.png){: width="40%"}
    
    ![Untitled](/assets/Images/2022-12-25-Future-Vehicle-Project/Untitled%204.png){: width="40%"}
    
    ![Untitled](/assets/Images/2022-12-25-Future-Vehicle-Project/Untitled%205.png){: width="40%"}
    

### 2차원 Map으로 변환

![Untitled](/assets/Images/2022-12-25-Future-Vehicle-Project/Untitled%206.png){: width="40%"}