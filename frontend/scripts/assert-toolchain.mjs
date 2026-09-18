#!/usr/bin/env node
/**
 * Toolchain assertion.
 *
 * Phases.md Phase 0: "Pinned frontend toolchain (React 18, Vite 5, Node 20,
 * pnpm 9) asserted in CI". Prompt.md §STACK: "exact versions pinned in
 * `package.json` + `engines`; CI asserts them".
 *
 * A caret range is not a pin: `^18.3.1` resolves to whatever is newest at
 * install time, so a green build says nothing about what shipped. This script
 * fails the build on three separate ways that guarantee can rot:
 *
 *   1. a declared version that is a range rather than an exact version,
 *   2. a major version that drifts from the one the spec names,
 *   3. an installed tree that disagrees with what `package.json` declares.
 */

import { existsSync, readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

/** Majors the spec names. Changing one of these needs an ADR, not an edit. */
const REQUIRED_MAJORS = {
  node: 20,
  pnpm: 9,
  react: 18,
  "react-dom": 18,
  vite: 5,
  tailwindcss: 3,
};

const EXACT_VERSION = /^\d+\.\d+\.\d+$/;

const failures = [];

function fail(message) {
  failures.push(message);
}

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function majorOf(version) {
  return Number.parseInt(version.split(".")[0], 10);
}

const pkg = readJson(resolve(projectRoot, "package.json"));
const declared = { ...pkg.dependencies, ...pkg.devDependencies };

// ─── 1. Every declared dependency is an exact version ────────────────────────

for (const [name, version] of Object.entries(declared)) {
  if (!EXACT_VERSION.test(version)) {
    fail(`${name} is declared as "${version}"; pin an exact version (no ^ or ~).`);
  }
}

// ─── 2. The majors the spec names ────────────────────────────────────────────

const nodeMajor = majorOf(process.versions.node);
if (nodeMajor !== REQUIRED_MAJORS.node) {
  fail(`Node ${REQUIRED_MAJORS.node}.x required, running ${process.versions.node}.`);
}

function detectPnpmVersion() {
  // Set by pnpm when this runs as a package script; falls back to asking pnpm.
  const agent = process.env.npm_config_user_agent ?? "";
  const fromAgent = /pnpm\/(\d+\.\d+\.\d+)/.exec(agent);
  if (fromAgent) {
    return fromAgent[1];
  }
  try {
    return execFileSync("pnpm", ["--version"], { encoding: "utf8", shell: true }).trim();
  } catch {
    return null;
  }
}

const pnpmVersion = detectPnpmVersion();
if (pnpmVersion === null) {
  fail(`pnpm ${REQUIRED_MAJORS.pnpm}.x required, but pnpm was not found.`);
} else if (majorOf(pnpmVersion) !== REQUIRED_MAJORS.pnpm) {
  fail(`pnpm ${REQUIRED_MAJORS.pnpm}.x required, running ${pnpmVersion}.`);
}

for (const name of ["react", "react-dom", "vite", "tailwindcss"]) {
  const version = declared[name];
  if (version === undefined) {
    fail(`${name} is not declared in package.json.`);
  } else if (majorOf(version) !== REQUIRED_MAJORS[name]) {
    fail(`${name} ${REQUIRED_MAJORS[name]}.x required, declared ${version}.`);
  }
}

// ─── 3. `engines` states the same constraint the CI runner enforces ──────────

for (const [field, major] of [
  ["node", REQUIRED_MAJORS.node],
  ["pnpm", REQUIRED_MAJORS.pnpm],
]) {
  const range = pkg.engines?.[field];
  if (typeof range !== "string") {
    fail(`package.json engines.${field} is missing.`);
  } else if (!range.includes(`${major}.`) || !range.includes(`${major + 1}.`)) {
    fail(`package.json engines.${field} is "${range}"; expected it to bound ${major}.x.`);
  }
}

const packageManager = pkg.packageManager ?? "";
if (!packageManager.startsWith(`pnpm@${REQUIRED_MAJORS.pnpm}.`)) {
  fail(`package.json packageManager is "${packageManager}"; expected pnpm@${REQUIRED_MAJORS.pnpm}.x.`);
}

// ─── 4. What is installed is what was declared ───────────────────────────────

const modulesDir = resolve(projectRoot, "node_modules");
if (existsSync(modulesDir)) {
  for (const [name, version] of Object.entries(declared)) {
    const manifest = resolve(modulesDir, name, "package.json");
    if (!existsSync(manifest)) {
      fail(`${name} is declared but not installed.`);
      continue;
    }
    const installed = readJson(manifest).version;
    if (installed !== version) {
      fail(`${name} is declared ${version} but ${installed} is installed.`);
    }
  }
} else {
  console.warn("node_modules not found: skipping the installed-tree check.");
}

// ─── Report ──────────────────────────────────────────────────────────────────

if (failures.length > 0) {
  console.error("Toolchain assertion failed:");
  for (const failure of failures) {
    console.error(`  - ${failure}`);
  }
  process.exit(1);
}

console.log(
  `Toolchain OK: node ${process.versions.node}, pnpm ${pnpmVersion}, ` +
    `react ${declared.react}, vite ${declared.vite}, tailwindcss ${declared.tailwindcss}.`,
);
