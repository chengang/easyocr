#include "webserver.hpp"
#include "webcontroller.hpp"

int WebServer::request_handler(void *cls, struct MHD_Connection *connection,
                               const char *url, const char *method,
                               const char *version, const char *upload_data,
                               size_t *upload_data_size, void **ptr) {

  std::cout << "Request: " << url << ", Method: " << method << std::endl;

  WebServer *server = (WebServer *)cls;

  Controller *controller = 0;
  for (int i = 0; i < server->controllers.size(); i++) {
    Controller *c = server->controllers.at(i);
    if (c->validPath(url, method)) {
      controller = c;
      break;
    }
  }

  if (!controller) {
    std::cout << "Path not found.\n";
    struct MHD_Response *response =
        MHD_create_response_from_buffer(0, 0, MHD_RESPMEM_PERSISTENT);
    return MHD_queue_response(connection, MHD_HTTP_NOT_FOUND, response);
  }

  return controller->handleRequest(connection, url, method, upload_data,
                                   upload_data_size);
}

WebServer::WebServer(int p) {
  port = p;
  daemon = 0;
}

void WebServer::addController(Controller *controller) {
  controllers.push_back(controller);
}

bool WebServer::start() {
  daemon = MHD_start_daemon(MHD_USE_THREAD_PER_CONNECTION, port, NULL, NULL,
                            &request_handler, this, MHD_OPTION_END);

  if (!daemon)
    return false;

  while (1) {
    sleep(10000);
  }

  MHD_stop_daemon(daemon);
  return true;
}
