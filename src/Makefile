all:
	clang++ main.cpp httpcurl.cpp easyocr.cpp webserver.cpp controller_ocr.cpp controller_deletefile.cpp helper.cpp priv_config.cpp -lboost_system -lboost_filesystem `pkg-config --cflags --libs opencv openssl libmicrohttpd libcurl` -o easyocr
test:
	./easyocr
install:
	install easyocr /bin
clean:
	rm -f ./easyocr

