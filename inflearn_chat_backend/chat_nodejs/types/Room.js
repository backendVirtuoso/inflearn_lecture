const { SocketLogger } = require("../logs/winston");

class Room {
    constructor() {
        this.forword = new Map();
        this.clients = new Set();
    }

    // TODO: 로그 추가
    join(client) {
        SocketLogger.info("new client");
        this.clients.add(client);
    }

    leave(client) {
        SocketLogger.info("removed client")
        this.clients.delete(client);
    }

    forwordMessage(message) {
        SocketLogger.info("send message to all client");
        for (const client of this.clients) {
            client.send(JSON.stringify(message));
        }
    }
}

function NewRoom() {
    return new Room();
}

module.exports = { NewRoom };