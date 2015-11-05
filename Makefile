all:
	clang++ main.cpp httpcurl.cpp easyocr.cpp webserver.cpp webcontroller.cpp helper.cpp -lboost_system -lboost_filesystem `pkg-config --cflags --libs opencv openssl libmicrohttpd libcurl` -o easyocr
test:
	./easyocr
install:
	install easyocr /bin
clean:
	rm -f ./easyocr

