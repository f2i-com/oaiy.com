//! Contacts to and from CSV files.
//!
//! **Export** (`GET /api/contacts/export.csv`): `name,number,notes,remembered`,
//! one contact a row, name-sorted; `remembered` is what the receptionist
//! remembered, joined with ` | `. UTF-8 with a byte-order mark (so Excel reads
//! it as UTF-8), CRLF line ends, RFC 4180 quoting, and a `'` before any cell a
//! spreadsheet would run as a formula (one starting `=`, `+`, `-`, `@`, a tab
//! or a carriage return): so an E.164 number is written `'+61491570006`.
//!
//! **Import** (`POST /api/contacts/import`): the common shapes, found from a
//! header row in the first ten (in any case): `name` / `full name`, or `first
//! name` + `last name`; `number` / `phone` / `mobile` / `phone number`;
//! `notes`; our own `remembered`; and a phone's export (Google Contacts'
//! `Name` or `First Name`…, `Phone 1 - Value` with its `Phone 1 - Label`, a
//! mobile preferred, and ` ::: `-joined values). With no header, a column of
//! phone numbers is looked for in the rows themselves. Quoted fields (with
//! commas, quotes and line breaks in them), a byte-order mark, and `;`- or
//! tab-separated files are read too, and the formula guard's `'` is taken off.
//! A local number (`0491 570 006`) is read as the chosen country's.
//!
//! Merging with the contacts there (the same key: the last nine digits):
//! the file's name replaces a name only when the person did not set it, or
//! when they ask for their names to be replaced; the file's notes are added
//! after the contact's own, unless they are there already; the name is then
//! the person's (`nameBy: owner`). A row is skipped for no number, one that is
//! not a phone's, a hidden one, the same number as an earlier row, or notes
//! past the limit. The whole import is one write.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use super::{check_number, clean_fact, clean_name, clean_notes, international, remember_fact, Book, By, Contact, Country, Error, MAX_FACT, MAX_NOTES};

/// The export's columns.
pub const HEADER: [&str; 4] = ["name", "number", "notes", "remembered"];
/// Between the facts in the `remembered` column.
pub const FACT_SEPARATOR: &str = " | ";
/// The most rows one import reads.
pub const MAX_ROWS: usize = 5000;
/// The largest import body (the file, in JSON).
pub const MAX_BYTES: usize = 8 * 1024 * 1024;
/// How many rows a report shows as they will go.
const SAMPLE: usize = 12;
/// How many skipped rows a report names.
const MAX_SKIPS: usize = 200;

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Every contact as the export's file.
pub fn export(contacts: &[Contact]) -> String {
    let mut out = String::from('\u{feff}');
    push_row(&mut out, &HEADER.map(String::from));
    for c in contacts {
        let number = if c.number.is_empty() { c.key.clone() } else { c.number.clone() };
        let facts = c.facts.iter().map(|f| f.text.as_str()).collect::<Vec<_>>().join(FACT_SEPARATOR);
        push_row(&mut out, &[c.name.clone(), number, c.notes.clone(), facts]);
    }
    out
}

fn push_row(out: &mut String, cells: &[String]) {
    out.push_str(&cells.iter().map(|c| cell(c)).collect::<Vec<_>>().join(","));
    out.push_str("\r\n");
}

/// What a spreadsheet runs as a formula when a cell starts with it.
const FORMULA: [char; 6] = ['=', '+', '-', '@', '\t', '\r'];

/// One cell, written: a `'` before a value a spreadsheet would run as a
/// formula; quoted when it holds a comma, semicolon, quote or line break, or
/// starts or ends with a space.
pub fn cell(value: &str) -> String {
    let guarded = if value.starts_with(FORMULA) { format!("'{value}") } else { value.to_string() };
    if guarded.contains([',', ';', '"', '\n', '\r']) || guarded.starts_with(' ') || guarded.ends_with(' ') {
        format!("\"{}\"", guarded.replace('"', "\"\""))
    } else {
        guarded
    }
}

/// A cell's value without the formula guard's `'`.
fn unguarded(value: &str) -> &str {
    match value.strip_prefix('\'') {
        Some(rest) if rest.starts_with(FORMULA) => rest,
        _ => value,
    }
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// The delimiter of a file: whichever of `,` `;` and tab its first lines
/// have most of outside quotes (a comma when none).
fn delimiter(text: &str) -> char {
    let mut counts = [(',', 0usize), (';', 0), ('\t', 0)];
    let (mut quoted, mut lines) = (false, 0);
    for ch in text.chars() {
        match ch {
            '"' => quoted = !quoted,
            '\n' if !quoted => {
                lines += 1;
                if lines >= 5 {
                    break;
                }
            }
            c if !quoted => {
                if let Some(slot) = counts.iter_mut().find(|(d, _)| *d == c) {
                    slot.1 += 1;
                }
            }
            _ => {}
        }
    }
    let mut best = counts[0];
    for c in counts {
        if c.1 > best.1 {
            best = c;
        }
    }
    best.0
}

/// A CSV file's records, each with the line it starts on (from 1). Blank lines are passed over.
pub fn parse(text: &str) -> Vec<(usize, Vec<String>)> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let delimiter = delimiter(text);
    let mut records = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let (mut line, mut start) = (1, 1);
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if quoted {
            match ch {
                '"' if chars.peek() == Some(&'"') => {
                    field.push('"');
                    chars.next();
                }
                '"' => quoted = false,
                c => {
                    if c == '\n' {
                        line += 1;
                    }
                    field.push(c);
                }
            }
            continue;
        }
        match ch {
            '"' if field.trim().is_empty() => {
                field.clear();
                quoted = true;
            }
            c if c == delimiter => row.push(std::mem::take(&mut field)),
            '\r' | '\n' => {
                if ch == '\r' && chars.peek() == Some(&'\n') {
                    chars.next();
                }
                row.push(std::mem::take(&mut field));
                if row.iter().any(|f| !f.trim().is_empty()) {
                    records.push((start, std::mem::take(&mut row)));
                } else {
                    row.clear();
                }
                line += 1;
                start = line;
            }
            c => field.push(c),
        }
    }
    row.push(field);
    if row.iter().any(|f| !f.trim().is_empty()) {
        records.push((start, row));
    }
    records
}

/// A heading as it is compared: lower case, words only.
fn heading(h: &str) -> String {
    let lower = h.trim().trim_start_matches('\u{feff}').to_lowercase();
    let words: String = lower.chars().map(|c| if c.is_alphanumeric() { c } else { ' ' }).collect();
    words.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Google Contacts' `Phone 1 - Value`: its number (`1`).
fn google_phone(h: &str) -> Option<&str> {
    let n = h.strip_prefix("phone ")?.strip_suffix(" value")?;
    (!n.is_empty() && n.chars().all(|c| c.is_ascii_digit())).then_some(n)
}

/// How much a column of numbers is preferred (0 first): a mobile, then a plain phone, a phone's export, then a home or work one.
fn phone_rank(h: &str) -> Option<u8> {
    match h {
        "mobile" | "mobile phone" | "mobile number" | "mobile no" | "mobile phone number" | "cell" | "cell phone" | "cellphone" | "cell number" | "mob" => Some(0),
        "number" | "phone" | "phone number" | "phone no" | "phonenumber" | "telephone" | "telephone number" | "tel" | "contact number" | "primary phone" | "main phone" => {
            Some(1)
        }
        "home phone" | "home phone 2" | "business phone" | "business phone 2" | "work phone" | "other phone" | "company main phone" | "car phone" => Some(3),
        _ if google_phone(h).is_some() => Some(2),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Phone {
    column: usize,
    /// Google's `Phone N - Label` beside it (Mobile, Home...).
    label: Option<usize>,
    rank: u8,
}

/// Which column holds what.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Columns {
    name: Option<usize>,
    first: Option<usize>,
    middle: Option<usize>,
    last: Option<usize>,
    numbers: Vec<Phone>,
    notes: Option<usize>,
    remembered: Option<usize>,
}

fn columns(header: &[String]) -> Columns {
    let names: Vec<String> = header.iter().map(|h| heading(h)).collect();
    let mut c = Columns::default();
    for (i, h) in names.iter().enumerate() {
        let slot = match h.as_str() {
            "name" | "full name" | "fullname" | "display name" | "contact name" | "contact" => Some(&mut c.name),
            "first name" | "firstname" | "given name" | "first" | "forename" => Some(&mut c.first),
            "middle name" | "additional name" | "middle" => Some(&mut c.middle),
            "last name" | "lastname" | "family name" | "surname" | "last" => Some(&mut c.last),
            "notes" | "note" | "comments" | "comment" => Some(&mut c.notes),
            "remembered" => Some(&mut c.remembered),
            _ => None,
        };
        if let Some(slot) = slot {
            slot.get_or_insert(i);
        }
        if let Some(rank) = phone_rank(h) {
            c.numbers.push(Phone { column: i, label: None, rank });
        }
    }
    for p in &mut c.numbers {
        if let Some(n) = google_phone(&names[p.column]) {
            let (label, kind) = (format!("phone {n} label"), format!("phone {n} type"));
            p.label = names.iter().position(|h| *h == label || *h == kind);
        }
    }
    c.numbers.sort_by_key(|p| (p.rank, p.column));
    c
}

/// With no header: a column most of whose first rows are phone numbers, and a column of names beside it.
fn guess(records: &[(usize, Vec<String>)]) -> Option<Columns> {
    let sample: Vec<&Vec<String>> = records.iter().take(20).map(|(_, r)| r).collect();
    let width = sample.iter().map(|r| r.len()).max()?;
    let is_number = |s: &String| !s.chars().any(char::is_alphabetic) && check_number(unguarded(s.trim())).is_ok();
    let count = |j: usize, f: &dyn Fn(&String) -> bool| sample.iter().filter(|r| r.get(j).is_some_and(f)).count();
    let number = (0..width).max_by_key(|&j| (count(j, &is_number), std::cmp::Reverse(j)))?;
    if count(number, &is_number) == 0 || count(number, &is_number) * 2 < sample.len() {
        return None;
    }
    let has_letters = |s: &String| s.chars().any(char::is_alphabetic);
    let name = (0..width).filter(|&j| j != number).find(|&j| count(j, &has_letters) * 2 >= sample.len());
    Some(Columns { name, numbers: vec![Phone { column: number, label: None, rank: 0 }], ..Columns::default() })
}

impl Columns {
    fn cell<'a>(&self, i: Option<usize>, r: &'a [String]) -> &'a str {
        i.and_then(|i| r.get(i)).map(|s| unguarded(s.trim())).unwrap_or("")
    }

    /// The row's name: its name column, else its first, middle and last names.
    fn name(&self, r: &[String]) -> String {
        let full = self.cell(self.name, r);
        if !full.trim().is_empty() {
            return clean_name(full);
        }
        clean_name(&[self.cell(self.first, r), self.cell(self.middle, r), self.cell(self.last, r)].join(" "))
    }

    /// The row's number: a phone's before one that is not, then a mobile, then by the columns' order.
    fn number(&self, r: &[String]) -> String {
        let mut best: Option<((bool, u8, usize), &str)> = None;
        for p in &self.numbers {
            let value = self.cell(Some(p.column), r).split(":::").map(str::trim).find(|v| !v.is_empty()).unwrap_or("");
            if value.is_empty() {
                continue;
            }
            let label = self.cell(p.label, r).to_lowercase();
            let rank = if label.contains("mobile") || label.contains("cell") { 0 } else { p.rank };
            let order = (check_number(value).is_err(), rank, p.column);
            if best.is_none_or(|(b, _)| order < b) {
                best = Some((order, value));
            }
        }
        best.map(|(_, v)| unguarded(v).to_string()).unwrap_or_default()
    }

    fn text(&self, i: Option<usize>, r: &[String]) -> String {
        self.cell(i, r).to_string()
    }

    /// The headings read, as the file writes them.
    fn used(&self, header: &[String]) -> Vec<String> {
        let mut at: Vec<usize> = [self.name, self.first, self.middle, self.last].into_iter().flatten().collect();
        at.extend(self.numbers.iter().map(|p| p.column));
        at.extend([self.notes, self.remembered].into_iter().flatten());
        at.into_iter().filter_map(|i| header.get(i)).map(|h| h.trim().trim_start_matches('\u{feff}').to_string()).collect()
    }
}

/// A file read for an import: where its header was, its columns, and the rows after it.
#[derive(Debug)]
pub struct Table {
    pub header_row: Option<usize>,
    pub columns: Columns,
    pub used: Vec<String>,
    pub rows: Vec<(usize, Vec<String>)>,
}

pub fn read_table(text: &str) -> Result<Table, Error> {
    let records = parse(text);
    if records.is_empty() {
        return Err(Error::new(400, "bad_csv", "the file has no rows"));
    }
    let too_many = |n: usize| Error::new(413, "too_many_rows", format!("the file has {n} rows: import at most {MAX_ROWS} at a time"));
    for (i, (line, fields)) in records.iter().enumerate().take(10) {
        let columns = columns(fields);
        if !columns.numbers.is_empty() {
            let rows = records[i + 1..].to_vec();
            if rows.len() > MAX_ROWS {
                return Err(too_many(rows.len()));
            }
            let used = columns.used(fields);
            return Ok(Table { header_row: Some(*line), columns, used, rows });
        }
    }
    if let Some(columns) = guess(&records) {
        if records.len() > MAX_ROWS {
            return Err(too_many(records.len()));
        }
        return Ok(Table { header_row: None, columns, used: Vec::new(), rows: records });
    }
    Err(Error::new(400, "bad_csv", "no column of phone numbers was found: name one number, phone or mobile in the header row"))
}

// ---------------------------------------------------------------------------
// Merging
// ---------------------------------------------------------------------------

/// `POST /api/contacts/import`.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImportRequest {
    /// The file's text.
    pub csv: String,
    /// Whose local numbers they are (AU when not given).
    #[serde(default)]
    pub country: Option<String>,
    /// Let the file's names replace the ones the person set.
    #[serde(default)]
    pub replace_names: bool,
    /// Only say what it would do.
    #[serde(default)]
    pub preview: bool,
}

/// What an import did, or would do.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportReport {
    pub preview: bool,
    pub country: &'static str,
    /// The line the header is on (none: the file has none).
    pub header_row: Option<usize>,
    /// The headings read.
    pub columns: Vec<String>,
    /// The rows with anything in them.
    pub rows: usize,
    pub added: usize,
    pub updated: usize,
    /// Already as the file has them.
    pub unchanged: usize,
    pub skipped: usize,
    /// How many were skipped for each reason: no_number, not_a_phone_number, hidden, duplicate, notes_too_long.
    pub reasons: BTreeMap<&'static str, usize>,
    /// The skipped rows (the first 200).
    pub skips: Vec<Row>,
    /// The first rows, as they go.
    pub sample: Vec<Row>,
}

/// One row of the file, as it goes.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Row {
    /// Its line in the file (from 1).
    pub row: usize,
    /// add, update, unchanged or skip.
    pub action: &'static str,
    pub name: String,
    pub number: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// Why it is skipped: a code...
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
    /// ...and words.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
    /// The person's own name, kept over the file's.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kept_name: Option<String>,
    /// Whether the file has notes for them.
    pub notes: bool,
}

/// The file's rows merged into `book` (the caller saves it, or not for a preview).
pub(crate) fn merge(book: &mut Book, table: &Table, country: &'static Country, replace_names: bool, now: &str) -> ImportReport {
    let mut report = ImportReport {
        preview: false,
        country: country.code,
        header_row: table.header_row,
        columns: table.used.clone(),
        rows: 0,
        added: 0,
        updated: 0,
        unchanged: 0,
        skipped: 0,
        reasons: BTreeMap::new(),
        skips: Vec::new(),
        sample: Vec::new(),
    };
    let mut seen: HashMap<String, usize> = HashMap::new();
    let cols = &table.columns;
    for (line, record) in &table.rows {
        let name = cols.name(record);
        let number = cols.number(record);
        let notes = cols.text(cols.notes, record);
        let remembered = cols.text(cols.remembered, record);
        if name.is_empty() && number.is_empty() && notes.trim().is_empty() && remembered.trim().is_empty() {
            continue;
        }
        report.rows += 1;
        let mut row = Row { row: *line, action: "skip", name: name.clone(), number: number.clone(), key: None, reason: None, why: None, kept_name: None, notes: !notes.trim().is_empty() };
        let skip = |report: &mut ImportReport, mut row: Row, reason: &'static str, why: String| {
            row.reason = Some(reason);
            row.why = Some(why);
            report.skipped += 1;
            *report.reasons.entry(reason).or_default() += 1;
            if report.sample.len() < SAMPLE {
                report.sample.push(row.clone());
            }
            if report.skips.len() < MAX_SKIPS {
                report.skips.push(row);
            }
        };
        let key = match check_number(&number) {
            Ok(k) => k,
            Err(why) => {
                skip(&mut report, row, why.code(), why.words().to_string());
                continue;
            }
        };
        row.key = Some(key.clone());
        if let Some(first) = seen.get(&key) {
            skip(&mut report, row, "duplicate", format!("the same number as row {first}"));
            continue;
        }
        seen.insert(key.clone(), *line);
        let Ok(notes) = clean_notes(&notes) else {
            skip(&mut report, row, "notes_too_long", format!("its notes are over {MAX_NOTES} characters"));
            continue;
        };
        let facts: Vec<String> = remembered
            .split(FACT_SEPARATOR)
            .filter_map(|f| clean_fact(&f.trim().chars().take(MAX_FACT).collect::<String>()).ok())
            .collect();
        let number = international(&number, country);
        match book.contacts.get_mut(&key) {
            None => {
                let mut c = Contact::new(&key, now);
                c.number = number;
                c.name.clone_from(&name);
                c.name_by = (!name.is_empty()).then_some(By::Owner);
                c.notes = notes;
                for f in &facts {
                    let _ = remember_fact(&mut c, f, By::Agent, now);
                }
                book.contacts.insert(key, c);
                row.action = "add";
                report.added += 1;
            }
            Some(c) => {
                // Every change worked out before any is made: notes too long leave the contact as it was.
                let joined = if notes.is_empty() || c.notes.contains(&notes) {
                    None
                } else if c.notes.is_empty() {
                    Some(notes)
                } else {
                    Some(format!("{}\n{notes}", c.notes))
                };
                if joined.as_ref().is_some_and(|j| j.chars().count() > MAX_NOTES) {
                    skip(&mut report, row, "notes_too_long", format!("its notes and the contact's are over {MAX_NOTES} characters together"));
                    continue;
                }
                let mut changed = false;
                if !name.is_empty() {
                    if c.named_by_owner() && !replace_names {
                        if c.name != name {
                            row.kept_name = Some(c.name.clone());
                        }
                    } else if c.name != name || c.name_by != Some(By::Owner) {
                        c.name.clone_from(&name);
                        c.name_by = Some(By::Owner);
                        changed = true;
                    }
                }
                if let Some(joined) = joined {
                    c.notes = joined;
                    changed = true;
                }
                if c.number.is_empty() {
                    c.number = number;
                    changed = true;
                }
                for f in &facts {
                    if let Ok((true, _)) = remember_fact(c, f, By::Agent, now) {
                        changed = true;
                    }
                }
                if changed {
                    c.updated_at = now.to_string();
                    row.action = "update";
                    report.updated += 1;
                } else {
                    row.action = "unchanged";
                    report.unchanged += 1;
                }
            }
        }
        if report.sample.len() < SAMPLE {
            report.sample.push(row);
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::super::tests::Dir;
    use super::super::{Change, Store};
    use super::*;

    fn import(s: &Store, csv: &str, replace_names: bool, preview: bool) -> ImportReport {
        s.import(&ImportRequest { csv: csv.to_string(), country: None, replace_names, preview }).unwrap()
    }

    fn names(s: &Store) -> Vec<(String, String, String)> {
        s.all().unwrap().into_iter().map(|c| (c.key, c.name, c.number)).collect()
    }

    #[test]
    fn a_cell_a_spreadsheet_would_run_is_guarded_and_quoting_is_proper() {
        assert_eq!(cell("=SUM(A1:A9)"), "'=SUM(A1:A9)");
        assert_eq!(cell("+61491570006"), "'+61491570006");
        assert_eq!(cell("-5"), "'-5");
        assert_eq!(cell("@liam"), "'@liam");
        assert_eq!(cell("\tx"), "'\tx");
        assert_eq!(cell("Liam"), "Liam");
        assert_eq!(cell("Smith, Liam"), "\"Smith, Liam\"");
        assert_eq!(cell("say \"hi\""), "\"say \"\"hi\"\"\"");
        assert_eq!(cell("two\nlines"), "\"two\nlines\"");
        assert_eq!(cell("a;b"), "\"a;b\"", "safe for a spreadsheet that splits on semicolons");
        assert_eq!(cell("=1,2"), "\"'=1,2\"", "guarded, then quoted");
        assert_eq!(cell(""), "");
        // Read back as it was.
        for v in ["=SUM(A1:A9)", "+61491570006", "-5", "@liam", "Smith, Liam", "say \"hi\"", "two\nlines", "=1,2", "it's", "'quoted'"] {
            let line = format!("{}\r\n", cell(v));
            let parsed = parse(&line);
            assert_eq!(unguarded(&parsed[0].1[0]), v, "{v:?} as {line:?}");
        }
    }

    #[test]
    fn the_export_is_utf8_with_a_bom_and_a_row_a_contact() {
        let d = Dir::new("export");
        let s = d.store();
        s.set("+61491570006", Change { name: Some("Liam".into()), number: Some("+61491570006".into()), notes: Some("Gate code: 1234, round the back".into()) }).unwrap();
        s.add_fact("0491570006", "Has a dog", By::Agent).unwrap();
        s.add_fact("0491570006", "Prefers mornings", By::Agent).unwrap();
        s.remember_name("0400000001", "Zoë").unwrap();
        s.set("0298765432", Change { name: Some("Anna".into()), ..Change::default() }).unwrap();
        let (csv, n) = s.export_csv().unwrap();
        assert_eq!(n, 3);
        assert!(csv.starts_with("\u{feff}name,number,notes,remembered\r\n"), "{csv:?}");
        let lines: Vec<&str> = csv.trim_start_matches('\u{feff}').split("\r\n").collect();
        assert_eq!(lines[1], "Anna,298765432,,", "no whole number known: the key");
        assert_eq!(lines[2], "Liam,'+61491570006,\"Gate code: 1234, round the back\",Has a dog | Prefers mornings");
        assert_eq!(lines[3], "Zoë,0400000001,,");
        assert_eq!(lines[4], "");
    }

    #[test]
    fn exported_then_imported_the_contacts_are_the_same() {
        let a = Dir::new("roundtrip-a");
        let sa = a.store();
        let tricky = "Line one, with a comma\n\"Quoted\" and =not a formula\n-dash first";
        sa.set("+61491570006", Change { name: Some("Liam".into()), number: Some("+61491570006".into()), notes: Some(tricky.into()) }).unwrap();
        sa.add_fact("+61491570006", "Has a dog, called Max", By::Agent).unwrap();
        sa.add_fact("+61491570006", "=2+2 is not run", By::Agent).unwrap();
        sa.set("+61298765432", Change { name: Some("=Anna; Bee".into()), number: Some("+61298765432".into()), ..Change::default() }).unwrap();
        sa.saw("+61411222333").unwrap();
        sa.remember_name("+61400000001", "Zoë").unwrap();
        let (csv, _) = sa.export_csv().unwrap();

        let b = Dir::new("roundtrip-b");
        let sb = b.store();
        let report = import(&sb, &csv, false, false);
        assert_eq!((report.added, report.updated, report.skipped), (4, 0, 0), "{report:?}");
        let shape = |s: &Store| {
            s.all().unwrap().into_iter().map(|c| (c.key, c.name, c.number, c.notes, c.facts.into_iter().map(|f| f.text).collect::<Vec<_>>())).collect::<Vec<_>>()
        };
        assert_eq!(shape(&sa), shape(&sb));
        // Imported names are the person's.
        assert_eq!(sb.get("0400000001").unwrap().name_by, Some(By::Owner));
        assert_eq!(sb.get("0411222333").unwrap().name_by, None, "no name, no one named them");
        // And again: nothing new.
        let again = import(&sb, &csv, false, false);
        assert_eq!((again.added, again.updated, again.unchanged), (0, 0, 4), "{again:?}");
    }

    #[test]
    fn the_common_header_shapes_are_read() {
        let d = Dir::new("shapes");
        let s = d.store();
        let files = [
            ("Name,Phone,Notes\r\nLiam,0491 570 006,Friend\r\n", "Liam"),
            ("first name,Last Name,MOBILE\nLiam,Smith,0491570006\n", "Liam Smith"),
            ("First Name,Last Name,Mobile\nLiam,,0491570006\n", "Liam"),
            ("\u{feff}Full Name;Phone Number;Notes\nLiam;0491 570 006;Friend\n", "Liam"),
            ("full name\tmobile number\nLiam\t0491570006\n", "Liam"),
            ("My contacts, exported\n\nName,Number\nLiam,+61 491 570 006\n", "Liam"),
            ("Liam,0491570006\nSam,0400000001\n", "Liam"),
        ];
        for (csv, want) in files {
            let _ = std::fs::remove_file(d.file());
            let r = import(&s, csv, false, false);
            assert_eq!(r.added, if csv.starts_with("Liam,") { 2 } else { 1 }, "{csv:?}: {r:?}");
            let c = s.get("0491570006").unwrap();
            assert_eq!((c.name.as_str(), c.number.as_str()), (want, "+61491570006"), "{csv:?}");
        }
        let r = import(&s, "My contacts, exported\n\nName,Number\nLiam,0491570006\n", false, true);
        assert_eq!(r.header_row, Some(3));
        assert_eq!(r.columns, ["Name", "Number"]);
        // A file with no numbers at all says so.
        let e = s.import(&ImportRequest { csv: "Name,Email\nLiam,l@x.au\n".into(), country: None, replace_names: false, preview: true }).unwrap_err();
        assert_eq!((e.status, e.code), (400, "bad_csv"));
        let e = s.import(&ImportRequest { csv: "Name,Phone\n".into(), country: Some("XX".into()), replace_names: false, preview: true }).unwrap_err();
        assert_eq!(e.code, "bad_country");
    }

    #[test]
    fn a_google_contacts_export_is_read_with_its_mobile_first() {
        let d = Dir::new("google");
        let s = d.store();
        // Google's older export: Name, Given Name..., Phone N - Type / Value, several numbers joined with :::.
        let older = "Name,Given Name,Additional Name,Family Name,Notes,Group Membership,Phone 1 - Type,Phone 1 - Value,Phone 2 - Type,Phone 2 - Value\n\
            Liam Smith,Liam,,Smith,\"Owns the café,\nopen Sundays\",* myContacts,Home,(02) 9876 5432,Mobile,0491 570 006 ::: 0400 000 009\n\
            ,Sam,,,,* myContacts,Mobile,,Work,0400 000 001\n";
        let r = import(&s, older, false, false);
        assert_eq!((r.added, r.skipped), (2, 0), "{r:?}");
        let liam = s.get("0491570006").unwrap();
        assert_eq!((liam.name.as_str(), liam.number.as_str()), ("Liam Smith", "+61491570006"), "the mobile, not the first number");
        assert_eq!(liam.notes, "Owns the café,\nopen Sundays");
        assert_eq!(s.get("0400000001").unwrap().name, "Sam", "Given Name when Name is empty");
        assert!(s.get("0298765432").is_err(), "one contact a row");

        // Google's newer export: First Name..., Phone 1 - Label / Value.
        let newer = "First Name,Middle Name,Last Name,Phonetic First Name,Nickname,Notes,Labels,E-mail 1 - Label,E-mail 1 - Value,Phone 1 - Label,Phone 1 - Value\n\
            Kim,J,Lee,,,,* myContacts,* Home,kim@example.com,Mobile,+61 411 222 333\n";
        let r = import(&s, newer, false, true);
        assert_eq!(r.added, 1);
        assert_eq!(r.columns, ["First Name", "Middle Name", "Last Name", "Phone 1 - Value", "Notes"]);
        assert_eq!((r.sample[0].name.as_str(), r.sample[0].number.as_str()), ("Kim J Lee", "+61 411 222 333"));
    }

    #[test]
    fn the_preview_counts_every_row_and_writes_nothing() {
        let d = Dir::new("preview");
        let s = d.store();
        s.set("0491570006", Change { name: Some("Liam".into()), notes: Some("Friend".into()), ..Change::default() }).unwrap();
        s.remember_name("0400000001", "Sam").unwrap();
        s.set("0400000002", Change { name: Some("Kim".into()), notes: Some("Tuesdays".into()), number: Some("+61400000002".into()) }).unwrap();
        let before = std::fs::read(d.file()).unwrap();
        let csv = "name,number,notes\n\
            Liam Smith,0491 570 006,Has a dog\n\
            Samuel,0400000001,\n\
            Kim,0400000002,Tuesdays\n\
            Jo,0400000003,\n\
            Jo again,+61 400 000 003,\n\
            Nobody,,\n\
            Short,12345,\n\
            Caller,Private,\n\
            ,,\n\
            Zed,0400000004,\n";
        let r = import(&s, csv, false, true);
        assert!(r.preview);
        assert_eq!(r.rows, 9, "the blank row is not one");
        assert_eq!((r.added, r.updated, r.unchanged, r.skipped), (2, 2, 1, 4), "{r:?}");
        assert_eq!(r.reasons, BTreeMap::from([("duplicate", 1), ("hidden", 1), ("no_number", 1), ("not_a_phone_number", 1)]));
        let why: Vec<(usize, &str)> = r.skips.iter().map(|x| (x.row, x.reason.unwrap())).collect();
        assert_eq!(why, [(6, "duplicate"), (7, "no_number"), (8, "not_a_phone_number"), (9, "hidden")]);
        assert_eq!(r.skips[0].why.as_deref(), Some("the same number as row 5"));
        let actions: Vec<(&str, &str)> = r.sample.iter().map(|x| (x.name.as_str(), x.action)).collect();
        assert_eq!(actions[..5], [("Liam Smith", "update"), ("Samuel", "update"), ("Kim", "unchanged"), ("Jo", "add"), ("Jo again", "skip")]);
        assert_eq!(r.sample[0].kept_name.as_deref(), Some("Liam"), "the person's name is kept");
        assert_eq!(std::fs::read(d.file()).unwrap(), before, "a preview writes nothing");

        // Imported: one write, the rules as the preview said.
        let r = import(&s, csv, false, false);
        assert_eq!((r.added, r.updated, r.unchanged, r.skipped), (2, 2, 1, 4));
        let liam = s.get("0491570006").unwrap();
        assert_eq!((liam.name.as_str(), liam.notes.as_str()), ("Liam", "Friend\nHas a dog"), "the name kept, the notes added after");
        let sam = s.get("0400000001").unwrap();
        assert_eq!((sam.name.as_str(), sam.name_by), ("Samuel", Some(By::Owner)), "the receptionist's name replaced, and the person's now");
        assert_eq!(s.get("0400000002").unwrap().notes, "Tuesdays", "the same notes are not added twice");
        assert_eq!(s.get("0400000003").unwrap().name, "Jo", "the first of a number wins");
        assert_eq!(names(&s).len(), 5);
    }

    #[test]
    fn replace_names_lets_the_file_rename_the_persons_contacts() {
        let d = Dir::new("replace");
        let s = d.store();
        s.set("0491570006", Change { name: Some("Liam".into()), ..Change::default() }).unwrap();
        let r = import(&s, "name,phone\nLiam Smith,0491570006\n", true, false);
        assert_eq!(r.updated, 1);
        assert_eq!(s.get("0491570006").unwrap().name, "Liam Smith");
        // Notes that would pass the limit together skip the row, leaving the contact as it was.
        s.set("0491570006", Change { notes: Some("n".repeat(MAX_NOTES - 5)), ..Change::default() }).unwrap();
        let r = import(&s, "name,phone,notes\nLiam Jones,0491570006,more than five\n", true, false);
        assert_eq!((r.skipped, r.reasons.get("notes_too_long")), (1, Some(&1)));
        assert_eq!(s.get("0491570006").unwrap().name, "Liam Smith");
    }

    #[test]
    fn a_countrys_local_numbers_are_read_as_its_own() {
        let d = Dir::new("country");
        let s = d.store();
        let r = s.import(&ImportRequest { csv: "name,phone\nAlex,(415) 555-0100\n".into(), country: Some("us".into()), replace_names: false, preview: false }).unwrap();
        assert_eq!(r.country, "US");
        assert_eq!(s.get("4155550100").unwrap().number, "+14155550100");
        // Without a country: Australian.
        import(&s, "name,phone\nLiam,0491 570 006\n", false, false);
        assert_eq!(s.get("0491570006").unwrap().number, "+61491570006");
    }

    #[test]
    fn quoted_fields_line_breaks_and_semicolons_are_read() {
        let recs = parse("\u{feff}a;b;c\r\n\"one; two\";\"line\r\nbreak\";\"say \"\"hi\"\"\"\r\n\r\nx;y;z");
        assert_eq!(recs.len(), 3);
        assert_eq!(recs[0].1, ["a", "b", "c"]);
        assert_eq!(recs[1].1, ["one; two", "line\r\nbreak", "say \"hi\""]);
        assert_eq!((recs[2].0, recs[2].1.clone()), (5, vec!["x".to_string(), "y".into(), "z".into()]), "the line it starts on");
        assert_eq!(parse("a,b\n1,2")[1].1, ["1", "2"], "no line end at the end");
        assert_eq!(parse("a\tb\n1\t2")[1].1, ["1", "2"]);
    }
}
