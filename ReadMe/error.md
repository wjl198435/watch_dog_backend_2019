## 1.stream:安装pip install av 失败解决办法:
    Could not find libavformat with pkg-config.
    Could not find libavcodec with pkg-config.
    Could not find libavdevice with pkg-config.
    Could not find libavutil with pkg-config.
    Could not find libavfilter with pkg-config.
    Could not find libswscale with pkg-config.
    Could not find libswresample with pkg-config.

解决办法:
    sudo apt install \
    autoconf \
    automake \
    build-essential \
    cmake \
    libass-dev \
    libfreetype6-dev \
    libjpeg-dev \
    libtheora-dev \
    libtool \
    libvorbis-dev \
    libx264-dev \
    pkg-config \
    wget \
    yasm \
    zlib1g-dev

    wget http://ffmpeg.org/releases/ffmpeg-3.2.tar.bz2
    tar -xjf ffmpeg-3.2.tar.bz2
    cd ffmpeg-3.2

    ./configure --disable-static --enable-shared --disable-doc
    make
    sudo make install

    pip install av

## 2.可选stream安装
    sudo apt-get install -y python-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libavresample-dev libavfilter-dev





### 3.解决Ubuntu16.04视频编码出现Unknown encoder 'libx264'问题
     3.1 sudo apt-get install libfdk-aac-dev libass-dev libopus-dev  libtheora-dev libvorbis-dev libvpx-dev libssl-dev
     3.2 sudo apt-get install nasm （版本大于2.13）
         下载： http://www.nasm.us/pub/nasm/releasebuilds/?C=M;O=D 
         cd ~/src/nasm-2.13.02
         ./configure
         make -j8
         sudo make install
     3.3 git clone git://git.videolan.org/x264.git
         cd x264
         ./configure --enable-static --enable-shared
         make -j8
         sudo make install
     3.4 
     git clone git://source.ffmpeg.org/ffmpeg.git ffmpeg
 ./configure  --enable-gpl   --enable-libass   --enable-libfdk-aac   --enable-libfreetype   --enable-libmp3lame   --enable-libopus   --enable-libtheora   --enable-libvorbis   --enable-libvpx   --enable-libx264   --enable-nonfree --enable-shared --enable-openssl 
     make -j8
     sudo make install
     export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
     ldd ffmpeg
     ffmpeg  –version
     
     
     
### Using USB webcams with Home Assistant
    https://www.home-assistant.io/blog/2016/06/23/usb-webcams-and-home-assistant/     
        
