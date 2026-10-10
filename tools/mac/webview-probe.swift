// tools/mac/webview-probe.swift: what a Mac's webview (WebKit) needs before it calls a page cross-origin isolated.
//
//     swiftc tools/mac/webview-probe.swift -o /tmp/webview-probe && /tmp/webview-probe [http://127.0.0.1:PORT/ ...]
//
// The Agent's code sandbox blocks on shared memory, which a page has only when it is cross-origin isolated. OAIY's
// window serves the Agent's page from a scheme of its own (`oaiy://localhost`) with the two policies that isolate a
// page in a browser, and on a Mac the page said it was not isolated. This asks WebKit itself: a page served from a
// scheme of its own, as the window serves one, under each embedder policy; and any http address given (a page a
// local server answers with the same headers), which is the other way a window could be given the page. For each it
// prints whether the page is isolated, has SharedArrayBuffer, is a secure context, and has the storage the Agent
// keeps its projects in.
import Cocoa
import WebKit

let report = "JSON.stringify({isolated: self.crossOriginIsolated, sharedArrayBuffer: typeof SharedArrayBuffer, secureContext: self.isSecureContext, storage: typeof (navigator.storage && navigator.storage.getDirectory)})"

final class Pages: NSObject, WKURLSchemeHandler {
    let policy: String
    init(policy: String) { self.policy = policy }

    func webView(_ webView: WKWebView, start task: WKURLSchemeTask) {
        let data = "<!doctype html><meta charset=utf-8><title>probe</title><p>probe</p>".data(using: .utf8)!
        var headers = [
            "Content-Type": "text/html; charset=utf-8",
            "Content-Length": String(data.count),
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Resource-Policy": "cross-origin",
        ]
        if policy != "none" { headers["Cross-Origin-Embedder-Policy"] = policy }
        let response = HTTPURLResponse(url: task.request.url!, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: headers)!
        task.didReceive(response)
        task.didReceive(data)
        task.didFinish()
    }

    func webView(_ webView: WKWebView, stop task: WKURLSchemeTask) {}
}

final class Probe: NSObject, WKNavigationDelegate {
    var views: [WKWebView] = []
    var names: [ObjectIdentifier: String] = [:]
    var left = 0

    func open(_ name: String, _ url: String, handler: (String, Pages)?) {
        let configuration = WKWebViewConfiguration()
        if let (scheme, pages) = handler { configuration.setURLSchemeHandler(pages, forURLScheme: scheme) }
        let view = WKWebView(frame: NSRect(x: 0, y: 0, width: 320, height: 200), configuration: configuration)
        view.navigationDelegate = self
        names[ObjectIdentifier(view)] = name
        views.append(view)
        left += 1
        view.load(URLRequest(url: URL(string: url)!))
    }

    func done(_ view: WKWebView, _ text: String) {
        print("\(names[ObjectIdentifier(view)] ?? "?"): \(text)")
        left -= 1
        if left == 0 { exit(0) }
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        webView.evaluateJavaScript(report) { value, error in
            self.done(webView, error.map { "could not ask the page: \($0.localizedDescription)" } ?? "\(value ?? "nothing")")
        }
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        done(webView, "did not load: \(error.localizedDescription)")
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        done(webView, "did not load: \(error.localizedDescription)")
    }
}

let application = NSApplication.shared
application.setActivationPolicy(.accessory)
let probe = Probe()
for (index, policy) in ["require-corp", "credentialless", "none"].enumerated() {
    let scheme = "oaiyprobe\(index)"
    probe.open("a scheme of its own, embedder policy \(policy)", "\(scheme)://localhost/index.html", handler: (scheme, Pages(policy: policy)))
}
for address in CommandLine.arguments.dropFirst() {
    probe.open("\(address) (what its server sends)", address, handler: nil)
}
DispatchQueue.main.asyncAfter(deadline: .now() + 40) {
    print("no answer from \(probe.left) page(s) in 40 s")
    exit(1)
}
application.run()
