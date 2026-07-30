/**
 * FFmpeg Utilities for Video/Audio Processing
 * 
 * Provides utility functions for common FFmpeg operations.
 * Requires FFmpeg to be bundled as a Tauri sidecar binary.
 */

// Minimal TauriAPI type for FFmpeg utils
interface TauriAPI {
    invoke<T>(command: string, args?: Record<string, unknown>): Promise<T>;
}

export interface FFmpegContext {
    tauri: TauriAPI;
    log: (level: 'info' | 'warn' | 'error', message: string) => void;
}

/**
 * Get the path to the bundled FFmpeg binary
 */
async function getFFmpegPath(ctx: FFmpegContext): Promise<string> {
    // Tauri sidecar binaries are accessed via Command
    // For now, we'll use shell command approach
    // In production, this would use Tauri's sidecar API

    // Check if ffmpeg is available in PATH or bundled location
    const appDataDir = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
    const ffmpegPath = `${appDataDir}/bin/ffmpeg`;

    // Fallback to system ffmpeg if bundled not found
    return ffmpegPath;
}

/**
 * Execute an FFmpeg command via Tauri shell
 */
async function executeFFmpeg(
    ctx: FFmpegContext,
    args: string[]
): Promise<{ success: boolean; stdout: string; stderr: string }> {
    try {
        const result = await ctx.tauri.invoke<{ code: number; stdout: string; stderr: string }>(
            'plugin:oaiy-filesystem|run_command',
            {
                command: 'ffmpeg',
                args: args,
                cwd: null
            }
        );

        return {
            success: result.code === 0,
            stdout: result.stdout,
            stderr: result.stderr
        };
    } catch (error) {
        ctx.log('error', `FFmpeg execution failed: ${error}`);
        return {
            success: false,
            stdout: '',
            stderr: String(error)
        };
    }
}

/**
 * Concatenate multiple video files into one
 * Uses FFmpeg concat demuxer for lossless concatenation of same-codec videos
 */
export async function concatVideos(
    ctx: FFmpegContext,
    inputPaths: string[],
    outputPath: string,
    options?: {
        transition?: 'none' | 'crossfade';
        transitionDuration?: number; // seconds
    }
): Promise<{ success: boolean; outputPath: string; error?: string }> {
    if (inputPaths.length === 0) {
        return { success: false, outputPath: '', error: 'No input videos provided' };
    }

    if (inputPaths.length === 1) {
        // Just copy the single video
        const result = await executeFFmpeg(ctx, [
            '-i', inputPaths[0],
            '-c', 'copy',
            '-y', outputPath
        ]);
        return { success: result.success, outputPath, error: result.stderr };
    }

    ctx.log('info', `[FFmpeg] Concatenating ${inputPaths.length} videos`);

    // Create a temporary concat list file
    const appDataDir = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
    const concatListPath = `${appDataDir}/temp/concat_${Date.now()}.txt`;

    // Build concat file content
    const concatContent = inputPaths.map(p => `file '${p.replace(/'/g, "'\\''")}'`).join('\n');

    // Write concat list
    await ctx.tauri.invoke('plugin:oaiy-filesystem|write_file', {
        path: concatListPath,
        content: concatContent,
        encoding: 'utf8'
    });

    try {
        // Execute FFmpeg concat
        const args = [
            '-f', 'concat',
            '-safe', '0',
            '-i', concatListPath,
            '-c', 'copy',
            '-y', outputPath
        ];

        const result = await executeFFmpeg(ctx, args);

        // Clean up temp file
        await ctx.tauri.invoke('plugin:oaiy-filesystem|delete_file', { path: concatListPath });

        if (result.success) {
            ctx.log('info', `[FFmpeg] Concatenation complete: ${outputPath}`);
        } else {
            ctx.log('error', `[FFmpeg] Concatenation failed: ${result.stderr}`);
        }

        return { success: result.success, outputPath, error: result.success ? undefined : result.stderr };
    } catch (error) {
        return { success: false, outputPath: '', error: String(error) };
    }
}

/**
 * Mix an audio track into a video with volume control
 */
export async function mixAudioIntoVideo(
    ctx: FFmpegContext,
    videoPath: string,
    audioPath: string,
    outputPath: string,
    options?: {
        videoVolume?: number; // 0.0 to 2.0, default 1.0
        audioVolume?: number; // 0.0 to 2.0, default 1.0
        replaceAudio?: boolean; // If true, replaces video audio entirely
    }
): Promise<{ success: boolean; outputPath: string; error?: string }> {
    const videoVol = options?.videoVolume ?? 1.0;
    const audioVol = options?.audioVolume ?? 1.0;
    const replaceAudio = options?.replaceAudio ?? false;

    ctx.log('info', `[FFmpeg] Mixing audio into video (videoVol=${videoVol}, audioVol=${audioVol})`);

    let args: string[];

    if (replaceAudio) {
        // Replace original audio with new audio
        args = [
            '-i', videoPath,
            '-i', audioPath,
            '-map', '0:v',      // Take video from first input
            '-map', '1:a',      // Take audio from second input
            '-c:v', 'copy',     // Copy video codec
            '-c:a', 'aac',      // Encode audio as AAC
            '-af', `volume=${audioVol}`,
            '-shortest',        // End when shortest stream ends
            '-y', outputPath
        ];
    } else {
        // Mix both audio tracks together - use duration=first to match video length
        args = [
            '-i', videoPath,
            '-i', audioPath,
            '-filter_complex', `[0:a]volume=${videoVol}[a0];[1:a]volume=${audioVol}[a1];[a0][a1]amix=inputs=2:duration=first[aout]`,
            '-map', '0:v',
            '-map', '[aout]',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-y', outputPath
        ];
    }

    const result = await executeFFmpeg(ctx, args);

    if (result.success) {
        ctx.log('info', `[FFmpeg] Audio mix complete: ${outputPath}`);
    } else {
        ctx.log('error', `[FFmpeg] Audio mix failed: ${result.stderr}`);
    }

    return { success: result.success, outputPath, error: result.success ? undefined : result.stderr };
}

/**
 * Extract audio from a video file
 */
export async function extractAudio(
    ctx: FFmpegContext,
    videoPath: string,
    outputPath: string,
    options?: {
        format?: 'mp3' | 'wav' | 'aac' | 'm4a';
        bitrate?: string; // e.g., '192k'
    }
): Promise<{ success: boolean; outputPath: string; error?: string }> {
    const format = options?.format ?? 'mp3';
    const bitrate = options?.bitrate ?? '192k';

    ctx.log('info', `[FFmpeg] Extracting audio to ${format}`);

    const args = [
        '-i', videoPath,
        '-vn',              // No video
        '-acodec', format === 'mp3' ? 'libmp3lame' : format === 'aac' ? 'aac' : 'pcm_s16le',
        '-ab', bitrate,
        '-y', outputPath
    ];

    const result = await executeFFmpeg(ctx, args);

    if (result.success) {
        ctx.log('info', `[FFmpeg] Audio extraction complete: ${outputPath}`);
    } else {
        ctx.log('error', `[FFmpeg] Audio extraction failed: ${result.stderr}`);
    }

    return { success: result.success, outputPath, error: result.success ? undefined : result.stderr };
}

/**
 * Get video/audio file information
 */
export async function getMediaInfo(
    ctx: FFmpegContext,
    filePath: string
): Promise<{
    duration?: number;
    width?: number;
    height?: number;
    fps?: number;
    hasAudio?: boolean;
    hasVideo?: boolean;
}> {
    const result = await executeFFmpeg(ctx, [
        '-i', filePath,
        '-hide_banner'
    ]);

    // Parse ffprobe-style output from stderr (ffmpeg outputs info to stderr)
    const output = result.stderr;

    const info: {
        duration?: number;
        width?: number;
        height?: number;
        fps?: number;
        hasAudio?: boolean;
        hasVideo?: boolean;
    } = {};

    // Parse duration
    const durationMatch = output.match(/Duration: (\d+):(\d+):(\d+\.\d+)/);
    if (durationMatch) {
        info.duration = parseInt(durationMatch[1]) * 3600 +
            parseInt(durationMatch[2]) * 60 +
            parseFloat(durationMatch[3]);
    }

    // Parse video stream
    const videoMatch = output.match(/Stream.*Video:.* (\d+)x(\d+).* (\d+(?:\.\d+)?) fps/);
    if (videoMatch) {
        info.width = parseInt(videoMatch[1]);
        info.height = parseInt(videoMatch[2]);
        info.fps = parseFloat(videoMatch[3]);
        info.hasVideo = true;
    }

    // Check for audio stream
    info.hasAudio = output.includes('Audio:');

    return info;
}
