import dearpygui.dearpygui as dpg
import cv2 as cv
import numpy as np
import matrix_detection
from models.configuration import Config


def convert_to_dpg(frame):
    data = frame.ravel()
    data = np.asfarray(data, dtype='f')
    texture_data = np.true_divide(data, 255.0)
    return texture_data


def main():

    config = Config.load_config()

    dpg.create_context()
    dpg.create_viewport(title='Custom Title', width=1000, height=800)
    dpg.setup_dearpygui()

    vid = cv.VideoCapture(0)
    ret, frame = vid.read()

    frame_width = vid.get(cv.CAP_PROP_FRAME_WIDTH)
    frame_height = vid.get(cv.CAP_PROP_FRAME_HEIGHT)

    # This popluates the default values for the windows
    data = np.flip(frame, 2)
    data = data.ravel()
    data = np.asfarray(data, dtype='f')
    texture_data = np.true_divide(data, 255.0)

    with dpg.texture_registry(show=True):
        dpg.add_raw_texture(frame_width, frame_height, texture_data,
                            tag="thresholded", format=dpg.mvFormat_Float_rgb)
        dpg.add_raw_texture(frame_width, frame_height, texture_data,
                            tag="final", format=dpg.mvFormat_Float_rgb)
        dpg.add_raw_texture(frame_width, frame_height, texture_data,
                            tag="blurred", format=dpg.mvFormat_Float_rgb)

    with dpg.window(label="Output Window", width=frame_width+50):
        dpg.add_text("Thresholded")
        dpg.add_image("thresholded")
        dpg.add_text("Blurred")
        dpg.add_image("blurred")
        dpg.add_text("Final")
        dpg.add_image("final")

    with dpg.window(label="Configuration", pos=(frame_width+50, 0)):
        dpg.add_slider_int(
            label="Threshold",
            default_value=config.threshold,
            width=100,
            max_value=255,
            min_value=0,
            tag="threshold_value"
        )
        dpg.add_checkbox(
            label="Adaptive thresholding",
            tag="adaptive_threshold"
        )
        dpg.add_slider_int(
            label="BlockSize",
            default_value=config.block_size,
            width=100,
            max_value=50,
            min_value=0,
            tag="block_size",
        )
        dpg.add_slider_int(
            label="C",
            default_value=config.c,
            width=100,
            max_value=50,
            min_value=0,
            tag="c",
        )
        dpg.add_slider_int(
            label="Blur size",
            default_value=config.blur_size,
            width=100,
            max_value=25,
            min_value=2,
            tag="blur_size",
        )

    dpg.show_metrics()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():

        ret, frame = vid.read()
        config.threshold = dpg.get_value("threshold_value")
        config.block_size = dpg.get_value("block_size") * 2 + 3
        config.adaptive_threshold = dpg.get_value("adaptive_threshold")
        config.c = dpg.get_value("c")
        config.blur_size = dpg.get_value("blur_size")*2+3

        decoded = matrix_detection.decode_matrix(frame, config)
        decoded = {k: convert_to_dpg(v) for k, v in decoded.items()}
        dpg.set_value("thresholded", decoded["thresholded"])
        dpg.set_value("final", decoded["final"])
        dpg.set_value("blurred", decoded["blurred"])
        dpg.render_dearpygui_frame()

    config.blur_size = (config.blur_size - 3)//2
    config.block_size = (config.block_size - 3)//2
    config.save_config()
    vid.release()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
