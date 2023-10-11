

# Vehicle Tracking using YOLOv8 and DeepSORT

This is a vehicle tracking project that uses YOLOv8 for object detection and DeepSORT for tracking the detected vehicles. The project also has a GUI built using Streamlit, which makes it easy to use and visualize the results.

<img src="img/of-2.gif">

## Usage

To use this project, you will need to have the following dependencies installed on your system:

- Python 3.6 or later
- YOLO v8
- DeepSORT
- PyTorch
- OpenCV
- Streamlit

Run the following command to start the Streamlit app:

```
streamlit run main.py
```

You can now open the app in your web browser at http://localhost:8501.

The app provides the following functionality:

- Detect Vehicle from image
- Track vehicles from video
- Count Vehicle (in/out) from video


## References

This project is using following resources:

- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [DeepSORT](https://github.com/nwojke/deep_sort)
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)