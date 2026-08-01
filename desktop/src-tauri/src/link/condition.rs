//! Deciding whether a trigger's condition holds.
//!
//! Bindings carry conditions written as JavaScript expressions, e.g.
//!
//! ```text
//! event && event.data ? Number(event.data.durationSeconds || 0) > 5 : false
//! ```
//!
//! Embedding a JavaScript engine to answer that would be a great deal of
//! attack surface for the shape of expression a trigger editor actually emits.
//! So this is a deliberately small language: literals, property paths, the two
//! coercions those expressions use, comparison, `&&`/`||`, `!`, parentheses and
//! the ternary. Anything outside it is **refused, not guessed**.
//!
//! # Which way it fails matters more than how often
//!
//! A condition that wrongly evaluates true sends the message, books the
//! appointment, calls the customer back. So an expression this cannot parse,
//! or one that reads a shape it does not understand, returns [`Verdict::Unknown`]
//! — and an unknown condition does not fire. The only path to firing is an
//! expression fully understood that evaluated truthy.
//!
//! # The semantics are JavaScript's, because the author wrote JavaScript
//!
//! `''`, `0` and `null` are falsy; `||` yields its first truthy operand rather
//! than a boolean; `String(null)` is `"null"`; `Number('')` is `0`. Getting any
//! of these subtly wrong would not error — it would fire the wrong flows, which
//! is the failure this module exists to prevent.

use serde_json::Value;

/// What a condition decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    True,
    False,
    /// Not understood. Never fires, and says why.
    Unknown(String),
}

impl Verdict {
    pub fn fires(&self) -> bool {
        matches!(self, Verdict::True)
    }
}

/// Evaluate `expr` against the event envelope bound to the name `event`.
pub fn evaluate(expr: &str, event: &Value) -> Verdict {
    let expr = expr.trim();
    if expr.is_empty() {
        // No condition is not a failed condition: it fires.
        return Verdict::True;
    }
    let mut p = Parser::new(expr);
    let value = match p.expression() {
        Ok(v) => v,
        Err(e) => return Verdict::Unknown(e),
    };
    if let Err(e) = p.expect_end() {
        return Verdict::Unknown(e);
    }
    match resolve(&value, event) {
        Ok(v) => {
            if truthy(&v) {
                Verdict::True
            } else {
                Verdict::False
            }
        }
        Err(e) => Verdict::Unknown(e),
    }
}

/// The expression tree. Small on purpose.
#[derive(Debug, Clone, PartialEq)]
enum Node {
    Literal(Value),
    /// A dotted path rooted at an identifier, e.g. `event.data.outcome`.
    Path(Vec<String>),
    Not(Box<Node>),
    Binary(&'static str, Box<Node>, Box<Node>),
    Ternary(Box<Node>, Box<Node>, Box<Node>),
    /// `String(x)` or `Number(x)` — the only two calls this understands.
    Coerce(Coercion, Box<Node>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Coercion {
    Str,
    Num,
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src: src.as_bytes(),
            pos: 0,
        }
    }

    fn expect_end(&mut self) -> Result<(), String> {
        self.skip_ws();
        if self.pos < self.src.len() {
            return Err(format!(
                "unexpected trailing input at byte {}: {:?}",
                self.pos,
                String::from_utf8_lossy(&self.src[self.pos..]).chars().take(24).collect::<String>()
            ));
        }
        Ok(())
    }

    fn skip_ws(&mut self) {
        while self.pos < self.src.len() && (self.src[self.pos] as char).is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.src.get(self.pos).copied()
    }

    fn eat(&mut self, token: &str) -> bool {
        self.skip_ws();
        if self.src[self.pos..].starts_with(token.as_bytes()) {
            self.pos += token.len();
            return true;
        }
        false
    }

    /// Ternary, the loosest binding form.
    fn expression(&mut self) -> Result<Node, String> {
        let cond = self.or()?;
        if self.eat("?") {
            let yes = self.expression()?;
            if !self.eat(":") {
                return Err("a ternary is missing its ':'".into());
            }
            let no = self.expression()?;
            return Ok(Node::Ternary(Box::new(cond), Box::new(yes), Box::new(no)));
        }
        Ok(cond)
    }

    fn or(&mut self) -> Result<Node, String> {
        let mut left = self.and()?;
        while self.eat("||") {
            let right = self.and()?;
            left = Node::Binary("||", Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn and(&mut self) -> Result<Node, String> {
        let mut left = self.comparison()?;
        while self.eat("&&") {
            let right = self.comparison()?;
            left = Node::Binary("&&", Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn comparison(&mut self) -> Result<Node, String> {
        let left = self.unary()?;
        // Longest first: `===` before `==`, `>=` before `>`.
        for op in ["===", "!==", "==", "!=", ">=", "<=", ">", "<"] {
            if self.eat(op) {
                let right = self.unary()?;
                let op: &'static str = match op {
                    "===" => "===",
                    "!==" => "!==",
                    "==" => "==",
                    "!=" => "!=",
                    ">=" => ">=",
                    "<=" => "<=",
                    ">" => ">",
                    _ => "<",
                };
                return Ok(Node::Binary(op, Box::new(left), Box::new(right)));
            }
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<Node, String> {
        self.skip_ws();
        // `!` but not `!=` — the comparison layer owns that.
        if self.peek() == Some(b'!') && self.src.get(self.pos + 1) != Some(&b'=') {
            self.pos += 1;
            return Ok(Node::Not(Box::new(self.unary()?)));
        }
        self.primary()
    }

    fn primary(&mut self) -> Result<Node, String> {
        self.skip_ws();
        let Some(c) = self.peek() else {
            return Err("the expression ended early".into());
        };
        if c == b'(' {
            self.pos += 1;
            let inner = self.expression()?;
            if !self.eat(")") {
                return Err("a '(' has no matching ')'".into());
            }
            return Ok(inner);
        }
        if c == b'\'' || c == b'"' {
            return self.string(c);
        }
        if c.is_ascii_digit() || (c == b'-' && self.src.get(self.pos + 1).is_some_and(u8::is_ascii_digit)) {
            return self.number();
        }
        if c.is_ascii_alphabetic() || c == b'_' || c == b'$' {
            return self.identifier_or_call();
        }
        Err(format!("unexpected character {:?}", c as char))
    }

    fn string(&mut self, quote: u8) -> Result<Node, String> {
        self.pos += 1;
        let mut out = String::new();
        while self.pos < self.src.len() {
            let c = self.src[self.pos];
            if c == b'\\' {
                // Only the escapes a trigger editor emits; an unknown one is a
                // refusal rather than a silently dropped backslash.
                let Some(next) = self.src.get(self.pos + 1).copied() else {
                    return Err("a string ends in a backslash".into());
                };
                out.push(match next {
                    b'n' => '\n',
                    b't' => '\t',
                    b'\\' => '\\',
                    b'\'' => '\'',
                    b'"' => '"',
                    other => return Err(format!("unsupported escape \\{}", other as char)),
                });
                self.pos += 2;
                continue;
            }
            if c == quote {
                self.pos += 1;
                return Ok(Node::Literal(Value::String(out)));
            }
            // Multi-byte characters pass through whole.
            let start = self.pos;
            let mut end = self.pos + 1;
            while end < self.src.len() && (self.src[end] & 0xC0) == 0x80 {
                end += 1;
            }
            out.push_str(&String::from_utf8_lossy(&self.src[start..end]));
            self.pos = end;
        }
        Err("a string has no closing quote".into())
    }

    fn number(&mut self) -> Result<Node, String> {
        let start = self.pos;
        if self.src[self.pos] == b'-' {
            self.pos += 1;
        }
        while self.pos < self.src.len()
            && (self.src[self.pos].is_ascii_digit() || self.src[self.pos] == b'.')
        {
            self.pos += 1;
        }
        let text = String::from_utf8_lossy(&self.src[start..self.pos]).to_string();
        text.parse::<f64>()
            .ok()
            .and_then(serde_json::Number::from_f64)
            .map(|n| Node::Literal(Value::Number(n)))
            .ok_or_else(|| format!("{text:?} is not a number"))
    }

    fn identifier_or_call(&mut self) -> Result<Node, String> {
        let start = self.pos;
        while self.pos < self.src.len()
            && (self.src[self.pos].is_ascii_alphanumeric()
                || self.src[self.pos] == b'_'
                || self.src[self.pos] == b'$')
        {
            self.pos += 1;
        }
        let word = String::from_utf8_lossy(&self.src[start..self.pos]).to_string();

        match word.as_str() {
            "true" => return Ok(Node::Literal(Value::Bool(true))),
            "false" => return Ok(Node::Literal(Value::Bool(false))),
            "null" | "undefined" => return Ok(Node::Literal(Value::Null)),
            _ => {}
        }

        if self.peek() == Some(b'(') {
            let coercion = match word.as_str() {
                "String" => Coercion::Str,
                "Number" => Coercion::Num,
                // Every other call is refused. Guessing what an unknown
                // function returns is how a condition fires wrongly.
                other => return Err(format!("the function {other}() is not available here")),
            };
            self.pos += 1;
            let inner = self.expression()?;
            if !self.eat(")") {
                return Err(format!("{word}( has no matching ')'"));
            }
            return Ok(Node::Coerce(coercion, Box::new(inner)));
        }

        let mut path = vec![word];
        loop {
            self.skip_ws();
            if self.src.get(self.pos) != Some(&b'.') {
                break;
            }
            self.pos += 1;
            let seg_start = self.pos;
            while self.pos < self.src.len()
                && (self.src[self.pos].is_ascii_alphanumeric() || self.src[self.pos] == b'_')
            {
                self.pos += 1;
            }
            if seg_start == self.pos {
                return Err("a '.' is not followed by a property name".into());
            }
            path.push(String::from_utf8_lossy(&self.src[seg_start..self.pos]).to_string());
        }
        Ok(Node::Path(path))
    }
}

/// A value during evaluation.
///
/// Not `serde_json::Value`, for one reason: JSON cannot represent NaN, and
/// `Number('abc')` must stay NaN all the way to the comparison. Collapsing it
/// to null read as ZERO, so `Number(x) >= 0` was true for text — a condition
/// firing on nonsense, which is the whole failure this module exists to avoid.
#[derive(Debug, Clone, PartialEq)]
enum Val {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    /// An array or object: truthy, and equal to nothing.
    Composite,
}

impl From<&Value> for Val {
    fn from(v: &Value) -> Self {
        match v {
            Value::Null => Val::Null,
            Value::Bool(b) => Val::Bool(*b),
            Value::Number(n) => n.as_f64().map(Val::Num).unwrap_or(Val::Null),
            Value::String(s) => Val::Str(s.clone()),
            Value::Array(_) | Value::Object(_) => Val::Composite,
        }
    }
}

/// Evaluate a parsed node against the event.
fn resolve(node: &Node, event: &Value) -> Result<Val, String> {
    match node {
        Node::Literal(v) => Ok(Val::from(v)),
        Node::Path(path) => {
            // Only `event` is in scope. A condition reading anything else is
            // reading something this desktop cannot supply, and answering
            // `undefined` would quietly change what it decides.
            if path[0] != "event" {
                return Err(format!("{:?} is not available to this condition", path[0]));
            }
            let mut cur = event;
            for segment in &path[1..] {
                match cur.get(segment) {
                    Some(v) => cur = v,
                    // A missing property is `undefined` in JavaScript too —
                    // this is the ordinary case, not a failure.
                    None => return Ok(Val::Null),
                }
            }
            Ok(Val::from(cur))
        }
        Node::Not(inner) => Ok(Val::Bool(!truthy(&resolve(inner, event)?))),
        Node::Ternary(cond, yes, no) => {
            if truthy(&resolve(cond, event)?) {
                resolve(yes, event)
            } else {
                resolve(no, event)
            }
        }
        Node::Coerce(kind, inner) => {
            let v = resolve(inner, event)?;
            Ok(match kind {
                Coercion::Str => Val::Str(to_string(&v)),
                // NaN is KEPT as NaN. It is falsy and compares false against
                // everything, which is exactly what JavaScript does.
                Coercion::Num => Val::Num(to_number(&v)),
            })
        }
        Node::Binary(op, left, right) => {
            let l = resolve(left, event)?;
            // Short-circuit, and yield the OPERAND, not a boolean: the whole
            // point of `event.data.outcome || ''` is the empty string.
            match *op {
                "||" => {
                    return if truthy(&l) { Ok(l) } else { resolve(right, event) };
                }
                "&&" => {
                    return if truthy(&l) { resolve(right, event) } else { Ok(l) };
                }
                _ => {}
            }
            let r = resolve(right, event)?;
            Ok(Val::Bool(match *op {
                "===" => strict_eq(&l, &r),
                "!==" => !strict_eq(&l, &r),
                "==" => loose_eq(&l, &r),
                "!=" => !loose_eq(&l, &r),
                _ => {
                    let (a, b) = (to_number(&l), to_number(&r));
                    // Every comparison against NaN is false, including `>=`.
                    // This is the line that matters: treating an unreadable
                    // number as 0 made `Number(x) >= 0` true for any text.
                    if a.is_nan() || b.is_nan() {
                        false
                    } else {
                        match *op {
                            ">" => a > b,
                            "<" => a < b,
                            ">=" => a >= b,
                            _ => a <= b,
                        }
                    }
                }
            }))
        }
    }
}

/// JavaScript truthiness.
fn truthy(v: &Val) -> bool {
    match v {
        Val::Null => false,
        Val::Bool(b) => *b,
        Val::Num(n) => *n != 0.0 && !n.is_nan(),
        Val::Str(s) => !s.is_empty(),
        // An object or array is truthy in JavaScript even when empty.
        Val::Composite => true,
    }
}

fn to_string(v: &Val) -> String {
    match v {
        Val::Null => "null".into(),
        Val::Bool(b) => b.to_string(),
        Val::Str(s) => s.clone(),
        Val::Num(n) => {
            if n.is_nan() {
                "NaN".into()
            } else if *n == n.trunc() && n.is_finite() {
                // JavaScript prints 5, not 5.0 — and these strings get compared
                // against literals an author typed.
                format!("{}", *n as i64)
            } else {
                n.to_string()
            }
        }
        Val::Composite => "[object Object]".into(),
    }
}

/// `Number(x)`, NaN and all.
fn to_number(v: &Val) -> f64 {
    match v {
        Val::Null => 0.0,
        Val::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Val::Num(n) => *n,
        Val::Str(s) => {
            let t = s.trim();
            if t.is_empty() {
                // Number('') is 0, not NaN. A duration read from an empty
                // string must not compare greater than anything.
                return 0.0;
            }
            t.parse::<f64>().unwrap_or(f64::NAN)
        }
        Val::Composite => f64::NAN,
    }
}

fn strict_eq(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Null, Val::Null) => true,
        (Val::Bool(x), Val::Bool(y)) => x == y,
        (Val::Str(x), Val::Str(y)) => x == y,
        // NaN is equal to nothing, itself included.
        (Val::Num(x), Val::Num(y)) => x == y,
        // Different types are never strictly equal — and two objects are equal
        // only by identity, which nothing here can have.
        _ => false,
    }
}

fn loose_eq(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Null, Val::Null) => true,
        (Val::Null, _) | (_, Val::Null) => false,
        (Val::Str(_), Val::Str(_)) | (Val::Bool(_), Val::Bool(_)) => strict_eq(a, b),
        _ => {
            let (x, y) = (to_number(a), to_number(b));
            !x.is_nan() && !y.is_nan() && x == y
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ev(data: Value) -> Value {
        json!({ "name": "aokie.call.ended", "data": data })
    }

    /// The conditions actually on this account's Aokie bindings, verbatim.
    #[test]
    fn the_real_conditions_decide_the_way_their_author_meant() {
        let duration = "event && event.data ? Number(event.data.durationSeconds || 0) > 5 : false";
        assert_eq!(evaluate(duration, &ev(json!({ "durationSeconds": 33 }))), Verdict::True);
        assert_eq!(evaluate(duration, &ev(json!({ "durationSeconds": 3 }))), Verdict::False);
        // Absent, null and empty all coerce to 0 — the reason the author wrote
        // `|| 0` at all, and each must stay BELOW the threshold.
        for missing in [json!({}), json!({ "durationSeconds": null }), json!({ "durationSeconds": "" })] {
            assert_eq!(evaluate(duration, &ev(missing)), Verdict::False);
        }
        // …and a numeric string still compares as a number.
        assert_eq!(evaluate(duration, &ev(json!({ "durationSeconds": "42" }))), Verdict::True);

        let outbound = "event && event.data ? String(event.data.direction || '') === 'outbound' : false";
        assert_eq!(evaluate(outbound, &ev(json!({ "direction": "outbound" }))), Verdict::True);
        assert_eq!(evaluate(outbound, &ev(json!({ "direction": "inbound" }))), Verdict::False);
        assert_eq!(evaluate(outbound, &ev(json!({}))), Verdict::False);

        let held = "event && event.data ? (String(event.data.outcome || '') === 'abandoned_on_hold' \
                    && String(event.data.direction || '') !== 'outbound') : false";
        assert_eq!(
            evaluate(held, &ev(json!({ "outcome": "abandoned_on_hold" }))),
            Verdict::True
        );
        assert_eq!(
            evaluate(held, &ev(json!({ "outcome": "abandoned_on_hold", "direction": "outbound" }))),
            Verdict::False
        );

        let missed = "event && event.data ? ((String(event.data.outcome || '') === 'missed' || \
                      String(event.data.status || '') === 'missed' || \
                      String(event.data.outcome || '') === 'abandoned_in_queue') && \
                      String(event.data.direction || '') !== 'outbound') : false";
        assert_eq!(evaluate(missed, &ev(json!({ "status": "missed" }))), Verdict::True);
        assert_eq!(evaluate(missed, &ev(json!({ "outcome": "abandoned_in_queue" }))), Verdict::True);
        assert_eq!(evaluate(missed, &ev(json!({ "outcome": "answered" }))), Verdict::False);

        let after = "event && event.data ? (Number(event.data.durationSeconds || 0) > 5 && \
                     String(event.data.outcome || '') !== 'missed' && \
                     String(event.data.status || '') !== 'missed' && \
                     String(event.data.outcome || '') !== 'terminated_abuse' && \
                     String(event.data.outcome || '') !== 'abandoned_on_hold' && \
                     String(event.data.outcome || '') !== 'abandoned_in_queue' && \
                     String(event.data.direction || '') !== 'outbound' && \
                     event.data.manager !== true) : false";
        assert_eq!(
            evaluate(after, &ev(json!({ "durationSeconds": 33, "outcome": "answered" }))),
            Verdict::True
        );
        assert_eq!(
            evaluate(after, &ev(json!({ "durationSeconds": 33, "outcome": "missed" }))),
            Verdict::False
        );
        assert_eq!(
            evaluate(after, &ev(json!({ "durationSeconds": 33, "manager": true }))),
            Verdict::False
        );
    }

    #[test]
    fn anything_not_understood_refuses_rather_than_guessing() {
        // A wrong `true` here sends the message, books the appointment, calls
        // the customer back. So every one of these must be Unknown, which does
        // not fire — not False, which would look like a considered decision.
        for expr in [
            "event.data.from.includes('+44')",          // a method call
            "event.data.tags[0] === 'vip'",             // indexing
            "(function () { return true })()",          // anything at all
            "event.data.x = 1",                         // assignment
            "event.data.n + 1 > 2",                     // arithmetic
            "Boolean(event.data.x)",                    // an unlisted function
            "event && event.data ?",                    // truncated
            "event.data.",                              // dangling dot
            "'unterminated",                            // unterminated string
            "other.thing === 1",                        // a root not in scope
        ] {
            match evaluate(expr, &ev(json!({ "from": "+4471", "n": 5 }))) {
                Verdict::Unknown(why) => assert!(!why.is_empty(), "{expr}"),
                other => panic!("{expr:?} must not be decided: got {other:?}"),
            }
        }
        // …and Unknown never fires.
        assert!(!Verdict::Unknown("x".into()).fires());
        assert!(!Verdict::False.fires());
        assert!(Verdict::True.fires());
    }

    #[test]
    fn or_yields_the_operand_not_a_boolean() {
        // `String(x || '')` depends on it: turned into a boolean, every
        // present value would stringify to "true" and no comparison would hold.
        let e = ev(json!({ "outcome": "answered" }));
        assert_eq!(
            evaluate("String(event.data.outcome || 'fallback') === 'answered'", &e),
            Verdict::True
        );
        assert_eq!(
            evaluate("String(event.data.missing || 'fallback') === 'fallback'", &e),
            Verdict::True
        );
        // And `&&` yields the falsy operand, which is what makes the guard
        // `event && event.data ? … : …` work when data is absent.
        assert_eq!(evaluate("event && event.data ? true : false", &json!({})), Verdict::False);
    }

    #[test]
    fn javascript_truthiness_is_used_because_the_author_wrote_javascript() {
        let e = ev(json!({
            "zero": 0, "empty": "", "no": false, "nil": null,
            "one": 1, "text": "x", "list": [], "obj": {}
        }));
        for falsy in ["zero", "empty", "no", "nil", "absent"] {
            assert_eq!(
                evaluate(&format!("event.data.{falsy} ? true : false"), &e),
                Verdict::False,
                "{falsy}"
            );
        }
        // An empty array or object is TRUTHY in JavaScript. Treating them as
        // empty-so-false is the kind of near-miss that fires the wrong flow.
        for truthy in ["one", "text", "list", "obj"] {
            assert_eq!(
                evaluate(&format!("event.data.{truthy} ? true : false"), &e),
                Verdict::True,
                "{truthy}"
            );
        }
    }

    #[test]
    fn strict_equality_does_not_cross_types() {
        let e = ev(json!({ "n": 5, "s": "5" }));
        assert_eq!(evaluate("event.data.n === 5", &e), Verdict::True);
        assert_eq!(evaluate("event.data.s === 5", &e), Verdict::False);
        assert_eq!(evaluate("event.data.s === '5'", &e), Verdict::True);
        // …but loose equality does, exactly as in JavaScript.
        assert_eq!(evaluate("event.data.s == 5", &e), Verdict::True);
        // A missing property is null/undefined, and only equals itself.
        assert_eq!(evaluate("event.data.gone === null", &e), Verdict::True);
        assert_eq!(evaluate("event.data.gone === ''", &e), Verdict::False);
    }

    #[test]
    fn an_empty_condition_fires_because_it_is_not_a_condition() {
        assert_eq!(evaluate("", &ev(json!({}))), Verdict::True);
        assert_eq!(evaluate("   ", &ev(json!({}))), Verdict::True);
    }

    #[test]
    fn a_number_never_compares_greater_than_something_unreadable() {
        // Number('abc') is NaN, and every comparison against NaN is false —
        // including `>`, which is the one that would otherwise fire a flow.
        let e = ev(json!({ "d": "not a number" }));
        assert_eq!(evaluate("Number(event.data.d) > 5", &e), Verdict::False);
        assert_eq!(evaluate("Number(event.data.d) < 5", &e), Verdict::False);
        assert_eq!(evaluate("Number(event.data.d) >= 0", &e), Verdict::False);
    }
}
