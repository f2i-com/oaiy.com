/*
 * Adapted from softn.com (apps/softn-studio/src/lib/providerConnection.ts),
 * Copyright f2i-com, licensed under the Apache License, Version 2.0.
 */

/**
 * Talking to a provider before any generation: where its endpoints are,
 * which models it offers, whether a key and an address work, and — when
 * they do not — a sentence a person can act on.
 *
 * The code lives in shared/providers/ now (endpoints.ts, errors.ts, models.ts), where the flow editor and the
 * providers origin use it too; this file keeps the Agent's imports as they were.
 */
export * from '@oaiy/shared/providers/endpoints';
export * from '@oaiy/shared/providers/errors';
export * from '@oaiy/shared/providers/models';
