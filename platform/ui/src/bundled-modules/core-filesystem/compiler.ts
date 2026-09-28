/**
 * Core Filesystem Module Compiler
 *
 * Compiles filesystem nodes (file_read, file_write, text_chunker) into JavaScript code.
 */

import type { ModuleCompiler, ModuleCompilerContext } from 'oaiy-core/src/module-types';

const CoreFilesystemCompiler: ModuleCompiler = {
  name: 'Filesystem',

  getNodeTypes() {
    return ['file_read', 'file_write', 'text_chunker'];
  },

  compileNode(nodeType: string, ctx: ModuleCompilerContext): string | null {
    const { node, inputs, outputVar, sanitizedId, skipVarDeclaration, escapeString, isInLoop, loopStartId, sanitizeId } = ctx;
    const data = node.data;
    const letOrAssign = skipVarDeclaration ? '' : 'let ';
    // The declared handle is 'text' — both text_chunker.json and
    // TextChunkerNode's inputHandles use that id, and the core compiler keys
    // this map strictly on `edge.targetHandle`. Reading only 'input' therefore
    // never matched, inputVar fell back to the literal 'null', and every Text
    // Chunker silently chunked nothing while the flow reported success.
    // 'input'/'default' stay as tolerant fallbacks for older saved graphs.
    const inputVar = inputs.get('text') || inputs.get('input') || inputs.get('default') || 'null';

    // Quote-aware CSV line parser, emitted once per CSV-parsing node and used
    // for BOTH the header row and data rows so quoted fields containing commas
    // map columns consistently. Returns an array of trimmed column strings.
    const csvLineParser = `
  function _parseCsvLine_${sanitizedId}(_line_) {
    let _cols_ = [];
    let _in_quote_ = false;
    let _col_ = "";
    for (let _c_ = 0; _c_ < _line_.length; _c_++) {
      let _char_ = _line_[_c_];
      if (_char_ === '"') {
        _in_quote_ = !_in_quote_;
      } else if (_char_ === ',' && !_in_quote_) {
        _cols_.push(_col_.trim());
        _col_ = "";
      } else {
        _col_ += _char_;
      }
    }
    _cols_.push(_col_.trim());
    return _cols_;
  }`;

    let code = `
  // --- Node: ${node.id} (${nodeType}) ---`;

    switch (nodeType) {
      case 'file_read': {
        const pathInput = inputs.get('path');
        const readAs = escapeString(String(data.readAs || 'text'));
        const csvHasHeader = data.csvHasHeader !== false;

        // For parsing modes (json, csv, lines), we need to read as text first
        const needsTextRead = ['json', 'csv', 'lines'].includes(readAs);
        const actualReadAs = needsTextRead ? 'text' : readAs;

        code += `
  // Get file path from input or property
  let _file_path_${sanitizedId} = ${pathInput || inputVar};
  // The 'hash' branch is a legacy check, harmless in V8.
  let _is_obj_${sanitizedId} = (typeof _file_path_${sanitizedId} === 'object' || typeof _file_path_${sanitizedId} === 'hash');
  if (_is_obj_${sanitizedId} && _file_path_${sanitizedId}.path) {
    _file_path_${sanitizedId} = _file_path_${sanitizedId}.path;
  }
  // Re-check after potentially extracting path
  _is_obj_${sanitizedId} = (typeof _file_path_${sanitizedId} === 'object' || typeof _file_path_${sanitizedId} === 'hash');
  if (_is_obj_${sanitizedId} && _file_path_${sanitizedId}.dataUrl) {
    // Already has content from file input
    ${letOrAssign}${outputVar} = _file_path_${sanitizedId}.dataUrl;
  } else {
    // Validate path before reading
    let _path_str_${sanitizedId} = "" + _file_path_${sanitizedId};
    // Skip invalid paths (not a real file path - must have : or / for Windows/Unix paths)
    if (_path_str_${sanitizedId}.length < 3 || (_path_str_${sanitizedId}.indexOf(":") < 0 && _path_str_${sanitizedId}.indexOf("/") < 0 && _path_str_${sanitizedId}.indexOf("\\\\") < 0)) {
      console.log("[FileRead] Skipping invalid path: " + _path_str_${sanitizedId});
      ${letOrAssign}${outputVar} = null;
    } else {
      ${letOrAssign}${outputVar} = await FileSystem.readFile(
        _path_str_${sanitizedId},
        "${actualReadAs}",
        "${node.id}"
      );
      if (${outputVar} === "__ABORT__") {
        console.log("[Workflow] aborted");
        return workflow_context;
      }

      // Check if result is a large file reference (for streaming with Text Chunker)
      let _is_file_ref_${sanitizedId} = false;
      if (typeof ${outputVar} === 'string' && ${outputVar}.indexOf('"__type":"file_ref"') >= 0) {
        _is_file_ref_${sanitizedId} = true;
        console.log("[FileRead] Large file detected - passing file reference for streaming");
      }

      // Only apply parsing if not a large file reference
      if (!_is_file_ref_${sanitizedId}) {`;

        // Add parsing logic for json, csv, lines (only for non-large files)
        if (readAs === 'json') {
          code += `
        // Parse JSON content
        if (${outputVar} != null) {
          try {
            ${outputVar} = JSON.parse(${outputVar});
          } catch (_parse_err_${sanitizedId}) {
            console.log("[FileRead] Failed to parse JSON: " + _parse_err_${sanitizedId});
            ${outputVar} = null;
          }
        }`;
        } else if (readAs === 'csv') {
          code += `
        // Parse CSV content
        if (${outputVar} != null) {${csvLineParser}
          let _csv_lines_${sanitizedId} = ${outputVar}.split("\\n").filter(function(_l_) { return _l_.trim().length > 0; });
          let _csv_headers_${sanitizedId} = [];
          let _csv_result_${sanitizedId} = [];
          let _csv_start_${sanitizedId} = 0;

          if (${csvHasHeader} && _csv_lines_${sanitizedId}.length > 0) {
            // Parse header row with the quote-aware parser (preserve outer-quote-strip + trim)
            _csv_headers_${sanitizedId} = _parseCsvLine_${sanitizedId}(_csv_lines_${sanitizedId}[0]).map(function(_h_) { return _h_.trim().replace(/^"|"$/g, ''); });
            _csv_start_${sanitizedId} = 1;
          }

          for (let _csv_i_${sanitizedId} = _csv_start_${sanitizedId}; _csv_i_${sanitizedId} < _csv_lines_${sanitizedId}.length; _csv_i_${sanitizedId}++) {
            let _csv_row_${sanitizedId} = _csv_lines_${sanitizedId}[_csv_i_${sanitizedId}];
            // Quote-aware CSV parsing (handles quoted fields with commas)
            let _csv_cols_${sanitizedId} = _parseCsvLine_${sanitizedId}(_csv_row_${sanitizedId});

            let _csv_row_obj_${sanitizedId} = JSON.parse("{}");
            _csv_row_obj_${sanitizedId}.index = _csv_i_${sanitizedId} - _csv_start_${sanitizedId};
            _csv_row_obj_${sanitizedId}.row = _csv_cols_${sanitizedId};
            if (_csv_headers_${sanitizedId}.length > 0) {
              // Create object with header keys
              let _csv_data_${sanitizedId} = JSON.parse("{}");
              for (let _csv_h_${sanitizedId} = 0; _csv_h_${sanitizedId} < _csv_headers_${sanitizedId}.length; _csv_h_${sanitizedId}++) {
                _csv_data_${sanitizedId}[_csv_headers_${sanitizedId}[_csv_h_${sanitizedId}]] = _csv_cols_${sanitizedId}[_csv_h_${sanitizedId}] || "";
              }
              _csv_row_obj_${sanitizedId}.data = _csv_data_${sanitizedId};
            }
            _csv_result_${sanitizedId}.push(_csv_row_obj_${sanitizedId});
          }
          ${outputVar} = _csv_result_${sanitizedId};
        }`;
        } else if (readAs === 'lines') {
          code += `
        // Parse lines content
        if (${outputVar} != null) {
          let _lines_arr_${sanitizedId} = ${outputVar}.split("\\n");
          let _lines_result_${sanitizedId} = [];
          for (let _lines_i_${sanitizedId} = 0; _lines_i_${sanitizedId} < _lines_arr_${sanitizedId}.length; _lines_i_${sanitizedId}++) {
            let _line_${sanitizedId} = _lines_arr_${sanitizedId}[_lines_i_${sanitizedId}];
            if (_line_${sanitizedId}.trim().length > 0) {
              _lines_result_${sanitizedId}.push(_line_${sanitizedId});
            }
          }
          ${outputVar} = _lines_result_${sanitizedId};
        }`;
        }

        code += `
      } // End of: if (!_is_file_ref)
    }
  }
  workflow_context["${node.id}"] = ${outputVar};`;
        break;
      }

      case 'file_write': {
        // Node settings
        const targetPath = escapeString(String(data.targetPath || data.path || ''));
        const filenamePattern = escapeString(String(data.filenamePattern || ''));
        const contentType = escapeString(String(data.contentType || 'text'));
        const createDirs = data.createDirectories !== false && data.createDirectory !== false;

        // Input handles
        const infoInput = inputs.get('info');      // FileInfo from folder listing (for template vars)
        const folderInput = inputs.get('folder');  // Folder path input
        const pathInput = inputs.get('path');      // Direct path input
        const contentInput = inputs.get('content') || inputs.get('default') || inputVar;

        code += `
  // Determine output path from inputs, settings, and filename pattern
  let _write_content_${sanitizedId} = ${contentInput};
  let _output_folder_${sanitizedId} = "${targetPath}";
  let _filename_pattern_${sanitizedId} = "${filenamePattern}";
  let _final_filename_${sanitizedId} = "";

  // Template variables from FileInfo (if connected)
  let _tpl_name_${sanitizedId} = "";
  let _tpl_nameWithoutExt_${sanitizedId} = "";
  let _tpl_ext_${sanitizedId} = "";
  let _tpl_index_${sanitizedId} = ${isInLoop && loopStartId ? `node_${sanitizeId(loopStartId)}_out_index != null ? node_${sanitizeId(loopStartId)}_out_index : 0` : '0'};
  let _source_dir_${sanitizedId} = "";`;

        // Get folder from folder input handle if connected
        if (folderInput) {
          code += `
  // Get folder from folder input
  let _folder_in_${sanitizedId} = ${folderInput};
  if (typeof _folder_in_${sanitizedId} === 'string' && _folder_in_${sanitizedId}.length > 0) {
    _output_folder_${sanitizedId} = _folder_in_${sanitizedId};
  }`;
        }

        // Extract template variables from FileInfo if connected
        if (infoInput) {
          code += `
  // Extract template variables from FileInfo object
  // Use JSON roundtrip to safely extract values and avoid VM stack issues
  let _info_json_${sanitizedId} = JSON.stringify(${infoInput});
  let _info_copy_${sanitizedId} = JSON.parse(_info_json_${sanitizedId});
  _tpl_name_${sanitizedId} = _info_copy_${sanitizedId}.name || "";
  _tpl_nameWithoutExt_${sanitizedId} = _info_copy_${sanitizedId}.nameWithoutExt || _info_copy_${sanitizedId}.name_without_ext || "";
  _tpl_ext_${sanitizedId} = _info_copy_${sanitizedId}.ext || _info_copy_${sanitizedId}.extension || "";

  // If no nameWithoutExt, derive from name using split instead of lastIndexOf
  if (!_tpl_nameWithoutExt_${sanitizedId} && _tpl_name_${sanitizedId}) {
    let _name_str_${sanitizedId} = "" + _tpl_name_${sanitizedId};
    let _dot_parts_${sanitizedId} = _name_str_${sanitizedId}.split(".");
    if (_dot_parts_${sanitizedId}.length > 1) {
      // Get extension (last part)
      let _ext_part_${sanitizedId} = _dot_parts_${sanitizedId}.pop();
      // Get name without extension (remaining parts joined)
      _tpl_nameWithoutExt_${sanitizedId} = _dot_parts_${sanitizedId}.join(".");
      if (!_tpl_ext_${sanitizedId}) {
        _tpl_ext_${sanitizedId} = _ext_part_${sanitizedId};
      }
    } else {
      _tpl_nameWithoutExt_${sanitizedId} = _name_str_${sanitizedId};
    }
  }

  // Get source directory from original path (fallback if no output folder specified)
  let _info_path_${sanitizedId} = "" + (_info_copy_${sanitizedId}.path || "");
  let _tpl_name_str_${sanitizedId} = "" + _tpl_name_${sanitizedId};
  if (_info_path_${sanitizedId}.length > 0 && _tpl_name_str_${sanitizedId}.length > 0) {
    // Use indexOf to find name position
    let _name_pos_${sanitizedId} = _info_path_${sanitizedId}.indexOf(_tpl_name_str_${sanitizedId});
    if (_name_pos_${sanitizedId} > 0) {
      _source_dir_${sanitizedId} = _info_path_${sanitizedId}.substring(0, _name_pos_${sanitizedId});
    }
  }`;
        }

        // Handle direct path input
        if (pathInput && !infoInput) {
          code += `
  // Get path from path input
  let _path_in_${sanitizedId} = ${pathInput};
  let _is_path_obj_${sanitizedId} = (typeof _path_in_${sanitizedId} === 'object' || typeof _path_in_${sanitizedId} === 'hash');
  if (_is_path_obj_${sanitizedId} && _path_in_${sanitizedId}.path) {
    // FileInfo-like object
    _tpl_name_${sanitizedId} = "" + (_path_in_${sanitizedId}.name || "");
    _tpl_nameWithoutExt_${sanitizedId} = "" + (_path_in_${sanitizedId}.nameWithoutExt || "");
    // Get source directory using indexOf instead of lastIndexOf
    let _pi_path_str_${sanitizedId} = "" + _path_in_${sanitizedId}.path;
    let _pi_name_str_${sanitizedId} = "" + _path_in_${sanitizedId}.name;
    let _pi_name_pos_${sanitizedId} = _pi_path_str_${sanitizedId}.indexOf(_pi_name_str_${sanitizedId});
    if (_pi_name_pos_${sanitizedId} > 0) {
      _source_dir_${sanitizedId} = _pi_path_str_${sanitizedId}.substring(0, _pi_name_pos_${sanitizedId});
    }
  } else if (typeof _path_in_${sanitizedId} === 'string') {
    // Direct path string - use as-is for output
    _output_folder_${sanitizedId} = "";
    _final_filename_${sanitizedId} = "" + _path_in_${sanitizedId};
  }`;
        }

        // Build final path
        code += `
  // Build final output path
  let _write_path_${sanitizedId} = "";

  // Get iteration index - use loop index directly with minimal variable usage
  // Keep logging minimal inside the loop body to reduce noise
  let _iter_num_${sanitizedId} = _tpl_index_${sanitizedId};
  if (typeof _iter_num_${sanitizedId} !== 'number' && typeof _iter_num_${sanitizedId} !== 'integer') {
    _iter_num_${sanitizedId} = 0;
  }

  if (_final_filename_${sanitizedId}) {
    // Direct path was provided
    _write_path_${sanitizedId} = _final_filename_${sanitizedId};
  } else {
    // Use output folder (from settings or folder input) or fall back to source directory
    let _use_folder_${sanitizedId} = _output_folder_${sanitizedId} || _source_dir_${sanitizedId};

    // Apply filename pattern or use default
    let _out_filename_${sanitizedId} = "";
    if (_filename_pattern_${sanitizedId}) {
      // Apply template substitutions to filename pattern
      _out_filename_${sanitizedId} = _filename_pattern_${sanitizedId};
      _out_filename_${sanitizedId} = _out_filename_${sanitizedId}.split("{{name}}").join(_tpl_name_${sanitizedId});
      _out_filename_${sanitizedId} = _out_filename_${sanitizedId}.split("{{nameWithoutExt}}").join(_tpl_nameWithoutExt_${sanitizedId});
      _out_filename_${sanitizedId} = _out_filename_${sanitizedId}.split("{{ext}}").join(_tpl_ext_${sanitizedId});
      _out_filename_${sanitizedId} = _out_filename_${sanitizedId}.split("{{index}}").join("" + _tpl_index_${sanitizedId});
    } else if (_tpl_nameWithoutExt_${sanitizedId}) {
      // Default: original name without extension + .txt
      _out_filename_${sanitizedId} = _tpl_nameWithoutExt_${sanitizedId} + ".txt";
    } else {
      // Fallback: generic output name
      _out_filename_${sanitizedId} = "output_" + _tpl_index_${sanitizedId} + ".txt";
    }

    // If no folder specified, get downloads path as fallback
    if (!_use_folder_${sanitizedId}) {
      _use_folder_${sanitizedId} = await FileSystem.getDownloadsPath();
    }

    // Build full path: folder + "/" + filename (FileSystem handles path normalization)
    _write_path_${sanitizedId} = "" + _use_folder_${sanitizedId} + "/" + _out_filename_${sanitizedId};
  }

  // Write the file - let FileSystem handle any path errors
  ${letOrAssign}${outputVar} = await FileSystem.writeFile(
      _write_path_${sanitizedId},
      _write_content_${sanitizedId},
      "${contentType}",
      ${createDirs},
      "${node.id}"
    );
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  workflow_context["${node.id}"] = ${outputVar};`;
        break;
      }

      case 'text_chunker': {
        // Guard chunkSize >= 1 and clamp overlap <= chunkSize - 1 so the raw
        // chunking loop's _pos always advances (avoids an infinite loop when
        // overlap >= chunkSize). These clamped values also feed
        // FileSystem.calculateFileChunks below.
        const chunkSize = Math.max(1, Number(data.chunkSize) || 1000);
        const overlap = Math.min(Number(data.overlap ?? 100), Math.max(0, chunkSize - 1));
        const contentType = escapeString(String(data.contentType || 'auto'));
        const csvHasHeader = data.csvHasHeader !== false;
        const indexName = escapeString(String(data.indexName || 'default'));

        code += `
  // Text chunker: split content based on type (auto-detect, raw, json_array, csv, lines)
  let _input_${sanitizedId} = ${inputVar};
  let _chunks_${sanitizedId} = [];
  let _content_type_${sanitizedId} = "${contentType}";
  let _is_file_ref_${sanitizedId} = false;

  // Check if input is a file reference (from large file read)
  if (typeof _input_${sanitizedId} === 'string' && _input_${sanitizedId}.indexOf('"__type":"file_ref"') >= 0) {
    try {
      let _ref_${sanitizedId} = JSON.parse(_input_${sanitizedId});
      if (_ref_${sanitizedId}.__type === "file_ref" && _ref_${sanitizedId}.path) {
        _is_file_ref_${sanitizedId} = true;
        console.log("[TextChunker] Detected large file reference: " + _ref_${sanitizedId}.path + " (" + _ref_${sanitizedId}.size + " bytes)");

        // Use streaming: calculate chunk boundaries then read each chunk
        let _chunk_refs_${sanitizedId} = await FileSystem.calculateFileChunks(
          _ref_${sanitizedId}.path,
          ${chunkSize},
          ${overlap},
          "${node.id}"
        );

        // Read each chunk content
        for (let _ci_${sanitizedId} = 0; _ci_${sanitizedId} < _chunk_refs_${sanitizedId}.length; _ci_${sanitizedId}++) {
          let _cref_${sanitizedId} = _chunk_refs_${sanitizedId}[_ci_${sanitizedId}];
          let _chunk_text_${sanitizedId} = await FileSystem.readChunkContent(
            _cref_${sanitizedId}.path,
            _cref_${sanitizedId}.start,
            _cref_${sanitizedId}.length,
            "text"
          );
          let _chunk_obj_${sanitizedId} = JSON.parse("{}");
          _chunk_obj_${sanitizedId}.text = _chunk_text_${sanitizedId};
          _chunk_obj_${sanitizedId}.index = _cref_${sanitizedId}.index;
          _chunk_obj_${sanitizedId}.start = _cref_${sanitizedId}.start;
          _chunk_obj_${sanitizedId}.end = _cref_${sanitizedId}.start + _cref_${sanitizedId}.length;
          _chunk_obj_${sanitizedId}.documentId = "${indexName}_" + _cref_${sanitizedId}.index;
          _chunk_obj_${sanitizedId}.total = _cref_${sanitizedId}.total;
          _chunks_${sanitizedId}.push(_chunk_obj_${sanitizedId});
        }
        console.log("[TextChunker] Streamed " + _chunks_${sanitizedId}.length + " chunks from large file");
      }
    } catch (_parse_err_${sanitizedId}) {
      console.log("[TextChunker] Not a valid file ref JSON, processing as text");
    }
  }

  // If not a file reference, process as in-memory text
  let _text_${sanitizedId} = "";
  if (!_is_file_ref_${sanitizedId}) {
    _text_${sanitizedId} = typeof _input_${sanitizedId} === 'string' ? _input_${sanitizedId} : JSON.stringify(_input_${sanitizedId} || '');

  // Auto-detect content type
  if (_content_type_${sanitizedId} === "auto") {
    let _trimmed_${sanitizedId} = _text_${sanitizedId}.trim();
    if (_trimmed_${sanitizedId}.startsWith("[") && _trimmed_${sanitizedId}.endsWith("]")) {
      _content_type_${sanitizedId} = "json_array";
    } else if (_trimmed_${sanitizedId}.startsWith("{") && _trimmed_${sanitizedId}.endsWith("}")) {
      // Single JSON object - wrap in array
      _content_type_${sanitizedId} = "json_array";
      _text_${sanitizedId} = "[" + _text_${sanitizedId} + "]";
    } else if (_trimmed_${sanitizedId}.indexOf(",") > 0 && _trimmed_${sanitizedId}.indexOf("\\n") > 0) {
      // Likely CSV (has commas and newlines)
      _content_type_${sanitizedId} = "csv";
    } else {
      _content_type_${sanitizedId} = "raw";
    }
  }

  if (_content_type_${sanitizedId} === "json_array") {
    // JSON Array: parse and return items (optionally batch them)
    try {
      let _arr_${sanitizedId} = JSON.parse(_text_${sanitizedId});
      if (Array.isArray(_arr_${sanitizedId})) {
        if (${chunkSize} > 1 && _arr_${sanitizedId}.length > ${chunkSize}) {
          // Batch into chunks of chunkSize items
          for (let _i_${sanitizedId} = 0; _i_${sanitizedId} < _arr_${sanitizedId}.length; _i_${sanitizedId} += ${chunkSize}) {
            let _batch_${sanitizedId} = _arr_${sanitizedId}.slice(_i_${sanitizedId}, _i_${sanitizedId} + ${chunkSize});
            let _chunk_obj_${sanitizedId} = JSON.parse("{}");
            _chunk_obj_${sanitizedId}.items = _batch_${sanitizedId};
            _chunk_obj_${sanitizedId}.index = Math.floor(_i_${sanitizedId} / ${chunkSize});
            _chunk_obj_${sanitizedId}.count = _batch_${sanitizedId}.length;
            _chunks_${sanitizedId}.push(_chunk_obj_${sanitizedId});
          }
        } else {
          // Return each item individually
          for (let _i_${sanitizedId} = 0; _i_${sanitizedId} < _arr_${sanitizedId}.length; _i_${sanitizedId}++) {
            let _chunk_obj_${sanitizedId} = JSON.parse("{}");
            _chunk_obj_${sanitizedId}.item = _arr_${sanitizedId}[_i_${sanitizedId}];
            _chunk_obj_${sanitizedId}.index = _i_${sanitizedId};
            _chunks_${sanitizedId}.push(_chunk_obj_${sanitizedId});
          }
        }
      }
    } catch(_e_${sanitizedId}) {
      console.log("[TextChunker] Failed to parse JSON: " + _e_${sanitizedId});
      // Fallback to raw text
      _content_type_${sanitizedId} = "raw";
    }
  }

  if (_content_type_${sanitizedId} === "csv") {
    // CSV: parse rows into objects (using header row as keys)${csvLineParser}
    let _lines_${sanitizedId} = _text_${sanitizedId}.split("\\n").filter(function(_l_) { return _l_.trim().length > 0; });
    let _headers_${sanitizedId} = [];
    let _start_row_${sanitizedId} = 0;

    if (${csvHasHeader} && _lines_${sanitizedId}.length > 0) {
      // Parse header row with the quote-aware parser (preserve outer-quote-strip + trim)
      _headers_${sanitizedId} = _parseCsvLine_${sanitizedId}(_lines_${sanitizedId}[0]).map(function(_h_) { return _h_.trim().replace(/^"|"$/g, ''); });
      _start_row_${sanitizedId} = 1;
    }

    for (let _i_${sanitizedId} = _start_row_${sanitizedId}; _i_${sanitizedId} < _lines_${sanitizedId}.length; _i_${sanitizedId}++) {
      let _row_${sanitizedId} = _lines_${sanitizedId}[_i_${sanitizedId}];
      // Quote-aware CSV parsing (handles quoted fields with commas)
      let _cols_${sanitizedId} = _parseCsvLine_${sanitizedId}(_row_${sanitizedId});

      let _chunk_obj_${sanitizedId} = JSON.parse("{}");
      _chunk_obj_${sanitizedId}.index = _i_${sanitizedId} - _start_row_${sanitizedId};
      _chunk_obj_${sanitizedId}.row = _cols_${sanitizedId};
      if (_headers_${sanitizedId}.length > 0) {
        // Create object with header keys
        let _data_${sanitizedId} = JSON.parse("{}");
        for (let _h_${sanitizedId} = 0; _h_${sanitizedId} < _headers_${sanitizedId}.length; _h_${sanitizedId}++) {
          _data_${sanitizedId}[_headers_${sanitizedId}[_h_${sanitizedId}]] = _cols_${sanitizedId}[_h_${sanitizedId}] || "";
        }
        _chunk_obj_${sanitizedId}.data = _data_${sanitizedId};
      }
      _chunks_${sanitizedId}.push(_chunk_obj_${sanitizedId});
    }
  }

  if (_content_type_${sanitizedId} === "lines") {
    // Lines: split by newlines
    let _lines_${sanitizedId} = _text_${sanitizedId}.split("\\n");
    for (let _i_${sanitizedId} = 0; _i_${sanitizedId} < _lines_${sanitizedId}.length; _i_${sanitizedId}++) {
      let _line_${sanitizedId} = _lines_${sanitizedId}[_i_${sanitizedId}];
      if (_line_${sanitizedId}.trim().length > 0) {
        let _chunk_obj_${sanitizedId} = JSON.parse("{}");
        _chunk_obj_${sanitizedId}.text = _line_${sanitizedId};
        _chunk_obj_${sanitizedId}.index = _i_${sanitizedId};
        _chunks_${sanitizedId}.push(_chunk_obj_${sanitizedId});
      }
    }
  }

  if (_content_type_${sanitizedId} === "raw" || _chunks_${sanitizedId}.length === 0) {
    // Raw text: split into overlapping character chunks
    let _pos_${sanitizedId} = 0;
    let _chunk_idx_${sanitizedId} = 0;
    while (_pos_${sanitizedId} < _text_${sanitizedId}.length) {
      let _end_${sanitizedId} = Math.min(_pos_${sanitizedId} + ${chunkSize}, _text_${sanitizedId}.length);
      let _chunk_${sanitizedId} = _text_${sanitizedId}.substring(_pos_${sanitizedId}, _end_${sanitizedId});
      let _chunk_obj_${sanitizedId} = JSON.parse("{}");
      _chunk_obj_${sanitizedId}.text = _chunk_${sanitizedId};
      _chunk_obj_${sanitizedId}.index = _chunk_idx_${sanitizedId};
      _chunk_obj_${sanitizedId}.start = _pos_${sanitizedId};
      _chunk_obj_${sanitizedId}.end = _end_${sanitizedId};
      _chunk_obj_${sanitizedId}.documentId = "${indexName}_" + _chunk_idx_${sanitizedId};
      _chunks_${sanitizedId}.push(_chunk_obj_${sanitizedId});
      _chunk_idx_${sanitizedId}++;
      _pos_${sanitizedId} = _end_${sanitizedId} - ${overlap};
      if (_pos_${sanitizedId} >= _text_${sanitizedId}.length - ${overlap}) break;
    }
  }

  console.log("[TextChunker] Detected type: " + _content_type_${sanitizedId} + ", chunks: " + _chunks_${sanitizedId}.length);
  } // End of: if (!_is_file_ref_${sanitizedId})
  ${letOrAssign}${outputVar} = _chunks_${sanitizedId};
  workflow_context["${node.id}"] = ${outputVar};`;
        break;
      }

      default:
        return null;
    }

    return code;
  },
};

export default CoreFilesystemCompiler;
