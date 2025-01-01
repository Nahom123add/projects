from flask import Flask, request, jsonify
import threading
import pjsua as pj

app = Flask(__name__)

# Online users (for simplicity, using a dictionary)
online_users = {}

# SIP Account Configuration
class MyAccountCallback(pj.AccountCallback):
    def __init__(self, account):
        pj.AccountCallback.__init__(self, account)

    def on_incoming_call(self, call):
        print("Incoming call from:", call.info().remote_uri)
        call.answer(200)

# Initialize SIP
class VoIPApp:
    def __init__(self):
        self.lib = pj.Lib()
        self.transport = None
        self.account = None

    def start(self, username, password, domain):
        try:
            self.lib.init(log_cfg=pj.LogConfig(level=3))
            self.transport = self.lib.create_transport(pj.TransportType.UDP, pj.TransportConfig(5060))
            self.lib.start()

            acc_cfg = pj.AccountConfig(domain, username, password)
            self.account = self.lib.create_account(acc_cfg)

            acc_cb = MyAccountCallback(self.account)
            self.account.set_callback(acc_cb)
            print("SIP account registered.")
        except pj.Error as e:
            print(f"Error initializing VoIP: {e}")

    def stop(self):
        if self.lib:
            self.lib.destroy()
            self.lib = None

voip_app = VoIPApp()

@app.route('/register', methods=['POST'])
def register_user():
    data = request.json
    username = data.get('username')
    ip_address = data.get('ip_address')

    if username and ip_address:
        online_users[username] = ip_address
        return jsonify({"message": "User registered successfully."}), 200
    return jsonify({"error": "Invalid data."}), 400

@app.route('/call', methods=['POST'])
def initiate_call():
    data = request.json
    caller = data.get('caller')
    callee = data.get('callee')

    if callee in online_users:
        callee_ip = online_users[callee]
        print(f"Initiating call from {caller} to {callee} at {callee_ip}.")
        # SIP call logic here (e.g., using pjsua API)
        return jsonify({"message": "Call initiated."}), 200
    return jsonify({"error": "Callee not available."}), 404

if __name__ == '__main__':
    # Start the SIP service
    threading.Thread(target=voip_app.start, args=("username", "password", "sipdomain.com")).start()

    # Run Flask server
    app.run(debug=True, host='0.0.0.0', port=5000)

    # Stop SIP service on exit
    voip_app.stop()
