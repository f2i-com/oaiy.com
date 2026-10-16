/*
 * The eGPU plugin's page: its setup's three parts (tinygrad's TinyGPU driver, the card, NVIDIA's compiler in Docker),
 * each with what it needs next, and the job the plugin runs for one of them (its step and what it said).
 *
 * It runs in OAIY Desktop's sandboxed plugin-screen frame, with no network: everything goes through
 * `window.PluginHost.command` to the plugin's connector (`setup.status`, `driver.install`, ...). In OAIY's setup
 * wizard (`PluginHost.setup`), it shows only the step's own card, and tells the wizard when that step is done.
 */
(function () {
  'use strict';

  var HOST = window.PluginHost;
  if (!HOST) {
    document.body.innerHTML = '<p class="eg-hint">Open this page in OAIY Desktop to set up the eGPU.</p>';
    return;
  }

  function $(id) { return document.getElementById(id); }

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function setHtml(el, html) {
    if (el.__egHtml === html) return;
    el.__egHtml = html;
    el.innerHTML = html;
  }

  function pill(id, kind, text) {
    var el = $(id);
    el.className = 'eg-pill' + (kind ? ' is-' + kind : '');
    el.querySelector('b').textContent = text;
  }

  function button(action, text, primary, disabled) {
    return '<button type="button" class="eg-button' + (primary ? ' is-primary' : '') + '" data-action="' + esc(action) + '"' +
      (disabled ? ' disabled' : '') + '>' + esc(text) + '</button>';
  }

  function facts(rows) {
    return '<dl class="eg-facts">' + rows.map(function (r) {
      return '<dt>' + esc(r[0]) + '</dt><dd>' + esc(r[1]) + '</dd>';
    }).join('') + '</dl>';
  }

  function errText(e) {
    var m = e && e.message ? e.message : String(e);
    if (/connector is not available|not started yet/i.test(m)) return 'The eGPU plugin is not running: start it on the Plugins page.';
    return m;
  }

  // ---- setup mode -------------------------------------------------------------------------------------------------

  var SETUP = HOST.setup && typeof HOST.setup.context === 'function' ? HOST.setup : null;
  var view = '';           // the wizard's step's view ('driver', 'card', 'compiler'), '' on the plugin's own page
  var doneSaid = false;

  function showOnly(which) {
    view = which;
    document.documentElement.classList.add('eg-setup');
    ['driver', 'card', 'compiler'].forEach(function (v) {
      $('eg-' + v).hidden = v !== which;
    });
  }

  // ---- the page ---------------------------------------------------------------------------------------------------

  var state = null;
  var busyAction = '';

  function renderDriver(s) {
    var d = s.driver || {};
    var job = s.job && s.job.name === 'driver' && s.job.state === 'running' ? s.job : null;
    var body, actions = '';
    if (d.enabled) {
      pill('eg-driver-pill', 'ok', 'Allowed');
      body = '<p>tinygrad\'s TinyGPU driver is installed and allowed: this Mac can talk to the card.</p>' +
        facts([['App', '/Applications/TinyGPU.app'], ['Driver', 'org.tinygrad.tinygpu, allowed']]);
    } else if (d.installed) {
      pill('eg-driver-pill', 'warn', 'Waiting for you');
      body = '<p>The driver is installed and waits for you to allow it. In System Settings, open <b>General</b>, ' +
        '<b>Login Items &amp; Extensions</b>, then <b>Driver Extensions</b> (the ⓘ beside it), and turn <b>TinyGPU</b> on. ' +
        'macOS may ask for your password.</p>' +
        '<p class="eg-hint">This page notices by itself once it is allowed.</p>';
      actions = button('driver.openSettings', 'Open System Settings', true) + button('driver.install', 'Ask macOS again', false, !!job);
    } else {
      pill('eg-driver-pill', 'warn', d.app ? 'Not asked for yet' : 'Not installed');
      body = '<p>' + (d.app
        ? 'TinyGPU.app is in Applications, but its driver has not been asked for yet.'
        : 'Install puts tinygrad\'s TinyGPU.app in Applications and asks macOS for its driver. macOS then asks you to allow it in System Settings, which opens for you.') +
        '</p><p class="eg-hint">TinyGPU is tinygrad\'s own driver for NVIDIA and AMD cards on a Mac, signed by tinygrad. It does nothing until a program opens the card.</p>';
      actions = button('driver.install', job ? 'Installing…' : 'Install the driver', true, !!job);
    }
    setHtml($('eg-driver-body'), body);
    setHtml($('eg-driver-actions'), actions);
    $('eg-driver').classList.toggle('is-done', !!d.enabled);
  }

  function renderCard(s) {
    var c = s.card || {};
    var pack = s.pack || {};
    var archs = (pack.archs || []).join(', ') || 'none';
    var body;
    if (!c.found) {
      pill('eg-card-pill', 'warn', 'Not found');
      body = '<p>No NVIDIA card is on this Mac\'s Thunderbolt yet. Connect the enclosure to a Thunderbolt port, turn it on, and give it a few seconds.</p>' +
        '<p class="eg-hint">This page looks again every few seconds. macOS lists what is connected in System Information, under PCI.</p>';
    } else if (!c.up) {
      pill('eg-card-pill', 'err', 'Link down');
      body = '<p>' + esc(c.name) + ' is there, but its link is down (' + esc(c.link || 'no link') + '). Turn the enclosure off, wait ten seconds, and turn it on again.</p>';
    } else {
      pill('eg-card-pill', 'ok', 'Connected');
      var kernels = c.covered
        ? (pack.kernels || 0).toLocaleString() + ' in the plugin for ' + c.arch
        : 'the plugin\'s are for ' + archs + (c.arch ? ', not ' + c.arch : '');
      body = '<p><b>' + esc(c.name) + '</b> is connected.</p>' +
        facts([['Link', (c.width || '?') + ' at ' + (c.speed || '?') + ' (' + (c.slot || 'Thunderbolt') + ')'], ['Architecture', c.arch || 'unknown'], ['Kernels', kernels]]);
      if (!c.covered) {
        body += '<p class="eg-hint is-warn">The plugin carries no kernels for this card\'s architecture: set up NVIDIA\'s compiler (below), and OAIY compiles them the first time each model needs them.</p>';
      }
    }
    if (s.missing > 0) {
      body += '<p class="eg-hint is-warn">' + s.missing + ' kernel' + (s.missing === 1 ? '' : 's') + ' a model asked for ' + (s.missing === 1 ? 'was' : 'were') +
        ' not in the plugin, with no compiler here to make ' + (s.missing === 1 ? 'it' : 'them') + '. NVIDIA\'s compiler (below) makes them; so does a newer plugin.</p>';
    }
    setHtml($('eg-card-body'), body);
    $('eg-card').classList.toggle('is-done', !!(c.found && c.up));
  }

  function renderCompiler(s) {
    var k = s.compiler || {};
    var d = s.docker || {};
    var job = s.job && s.job.name === 'compiler' && s.job.state === 'running' ? s.job : null;
    var body, actions = '';
    var why = '<p class="eg-hint">The plugin carries the kernels OAIY\'s models use, compiled ahead of time. A model newer than the plugin may need more: NVIDIA\'s compiler (nvcc) makes them, and on a Mac it runs in Docker. The first time each is needed takes a few seconds; after that it is kept.</p>';
    if (k.ready) {
      pill('eg-compiler-pill', 'ok', 'Ready');
      body = '<p>' + (k.native ? 'nvcc is on this Mac\'s PATH.' : 'nvcc in <span class="eg-path">~/.local/bin</span> runs NVIDIA\'s compiler in Docker.') + '</p>';
      if (!k.native && !d.running) {
        body += '<p class="eg-hint is-warn">Docker Desktop is not running now. It only needs to be when a model needs a kernel the plugin does not carry.</p>';
        actions = button('docker.start', 'Start Docker', false);
      }
    } else if (!d.cli && !d.app) {
      pill('eg-compiler-pill', '', 'Optional');
      body = why + '<p>Docker Desktop is not installed. Get it from docker.com, open it once, then come back here.</p>';
      actions = button('docker.download', 'Get Docker Desktop', true) + button('compiler.install', 'Set up the compiler', false, true);
    } else {
      pill('eg-compiler-pill', job ? 'warn' : '', job ? 'Setting up' : 'Optional');
      body = why + '<p>Set up builds tinygrad\'s compiler image in Docker (about 2 GB to download, a few minutes) and puts nvcc in <span class="eg-path">~/.local/bin</span>' +
        (d.running ? '.' : '. Docker Desktop starts first.') + '</p>';
      actions = button('compiler.install', job ? 'Setting up…' : 'Set up the compiler', true, !!job) + (d.running ? '' : button('docker.start', 'Start Docker', false));
    }
    setHtml($('eg-compiler-body'), body);
    setHtml($('eg-compiler-actions'), actions);
    $('eg-compiler').classList.toggle('is-done', !!k.ready);
  }

  function renderJob(s) {
    var j = s.job;
    // (the last job, running or how it ended; in the wizard, only its own step's)
    var show = !!j && (!view || view === j.name);
    $('eg-job').hidden = !show;
    if (!show) return;
    var names = { driver: 'Installing the TinyGPU driver', compiler: 'Setting up NVIDIA\'s compiler' };
    $('eg-job-title').textContent = names[j.name] || j.name;
    if (j.state === 'running') pill('eg-job-pill', 'warn is-busy', 'Running · ' + j.seconds + ' s');
    else if (j.state === 'done') pill('eg-job-pill', 'ok', 'Done');
    else pill('eg-job-pill', 'err', 'Failed');
    $('eg-job-step').textContent = j.state === 'failed' ? (j.error || 'It failed.') : j.step || '';
    var log = $('eg-job-log');
    var text = (j.log || []).join('\n');
    if (log.textContent !== text) {
      var atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
      log.textContent = text;
      if (atEnd) log.scrollTop = log.scrollHeight;
    }
  }

  function render(s) {
    state = s;
    renderDriver(s);
    renderCard(s);
    renderCompiler(s);
    renderJob(s);
    var steps = s.steps || {};
    var c = s.card || {};
    var ready = steps.driver && steps.card && (c.covered || steps.compiler);
    pill('eg-overall', ready ? 'ok' : 'warn', ready ? 'Ready' : 'Setup needed');
    $('eg-foot').textContent = 'eGPU plugin ' + (s.version || '') + ' · tinygrad ' + String(s.tinygrad || '?').slice(0, 9) + ' · ' +
      ((s.pack || {}).kernels || 0).toLocaleString() + ' kernels for ' + (((s.pack || {}).archs || []).join(', ') || 'no card');
    if (SETUP && view && !doneSaid && steps[view]) {
      doneSaid = true;
      Promise.resolve(SETUP.done()).catch(function () {});
    } else if (view && !steps[view]) {
      doneSaid = false;
    }
  }

  function renderError(e) {
    ['driver', 'card', 'compiler'].forEach(function (v) {
      pill('eg-' + v + '-pill', 'err', 'Unknown');
      setHtml($('eg-' + v + '-body'), '<p class="eg-hint is-err">' + esc(errText(e)) + '</p>');
    });
    pill('eg-overall', 'err', 'Not running');
  }

  // ---- polling: every 3 s, every second while a job runs, not while hidden ----------------------------------------

  var timer = null;
  var inFlight = false;

  function poll() {
    if (inFlight) return;
    inFlight = true;
    Promise.resolve(HOST.command('setup.status'))
      .then(render, renderError)
      .then(function () {
        inFlight = false;
        schedule();
      });
  }

  function schedule() {
    clearTimeout(timer);
    if (document.hidden) return;
    var running = state && state.job && state.job.state === 'running';
    timer = setTimeout(poll, running ? 1000 : 3000);
  }

  document.addEventListener('visibilitychange', function () {
    if (!document.hidden) poll();
  });

  // ---- buttons ------------------------------------------------------------------------------------------------------

  var SAID = {
    'driver.install': 'Installing the TinyGPU driver: macOS will ask you to allow it.',
    'compiler.install': 'Setting up NVIDIA\'s compiler in Docker: this takes a few minutes.',
    'driver.openSettings': 'System Settings is open: General, Login Items & Extensions, Driver Extensions.',
    'docker.download': 'Docker Desktop\'s page is open in your browser.',
    'docker.start': 'Starting Docker Desktop.'
  };

  document.addEventListener('click', function (ev) {
    var b = ev.target && ev.target.closest ? ev.target.closest('button[data-action]') : null;
    if (!b || b.disabled || busyAction) return;
    var action = b.getAttribute('data-action');
    busyAction = action;
    b.disabled = true;
    Promise.resolve(HOST.command(action))
      .then(function (res) {
        if (res && res.started === false) {
          HOST.toast('info', 'Wait for the job running now (' + res.busy + ') to end first.');
        } else if (res && res.opened === false) {
          HOST.toast('error', res.error || 'It could not be opened.');
        } else if (SAID[action]) {
          HOST.toast('info', SAID[action]);
        }
      }, function (e) {
        HOST.toast('error', errText(e));
      })
      .then(function () {
        busyAction = '';
        b.disabled = false;
        poll();
      });
  });

  // ---- start --------------------------------------------------------------------------------------------------------

  if (SETUP) {
    Promise.resolve(SETUP.context()).then(function (ctx) {
      if (ctx && ctx.view) showOnly(ctx.view);
      poll();
    }, poll);
  } else {
    poll();
  }
})();
