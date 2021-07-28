FROM geodels/vearth:latest

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir \
		ipyvtklink \
		numba \
		imageio_ffmpeg
        