/**
 * Files picked or dropped for an incident upload, with the folder path each
 * came from. A whole incident folder can be uploaded at once: the folder
 * names say which relay a record came from ("CBF B DIAMETER 2_/DR/…"), and
 * two relays often name their records alike, so the backend pairs .cfg/.dat
 * per folder and keeps the path.
 */

const relativePaths = new WeakMap<File, string>();

/** Files the incident upload takes: COMTRADE (.cfg/.dat/.cff) and MiCOM event lists (.evt). */
export const INCIDENT_UPLOAD_SUFFIXES = [".cfg", ".dat", ".cff", ".evt"];

/** The name to send for a file: its folder path when it came from a folder. */
export function uploadPath(file: File): string {
  const picked = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
  return relativePaths.get(file) || picked || file.name;
}

export function fileSuffix(name: string): string {
  const base = name.split(/[\\/]/).pop() || name;
  const dot = base.lastIndexOf(".");
  return dot >= 0 ? base.slice(dot).toLowerCase() : "";
}

/** Keep the files an incident upload takes; a folder also holds photos, settings and archives. */
export function keepSupported(files: File[]): { accepted: File[]; skipped: number } {
  const accepted = files.filter((f) => INCIDENT_UPLOAD_SUFFIXES.includes(fileSuffix(uploadPath(f))));
  return { accepted, skipped: files.length - accepted.length };
}

/**
 * Every file of a drop, folders walked recursively. The entries are taken
 * while the drop event is live (the browser empties the list after it), then
 * read asynchronously.
 */
export async function filesFromDrop(dataTransfer: DataTransfer): Promise<File[]> {
  const items = Array.from(dataTransfer.items || []);
  const entries = items
    .map((item) => (typeof item.webkitGetAsEntry === "function" ? item.webkitGetAsEntry() : null))
    .filter((entry): entry is FileSystemEntry => entry !== null);
  if (!entries.some((entry) => entry.isDirectory)) return Array.from(dataTransfer.files || []);
  const files: File[] = [];
  await Promise.all(entries.map((entry) => walk(entry, "", files)));
  return files;
}

async function walk(entry: FileSystemEntry, prefix: string, out: File[]): Promise<void> {
  const path = prefix ? `${prefix}/${entry.name}` : entry.name;
  if (entry.isFile) {
    const file = await new Promise<File>((resolve, reject) => (entry as FileSystemFileEntry).file(resolve, reject));
    relativePaths.set(file, path);
    out.push(file);
    return;
  }
  const reader = (entry as FileSystemDirectoryEntry).createReader();
  for (;;) {
    // readEntries hands out a directory in batches until it returns none.
    const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject));
    if (batch.length === 0) break;
    await Promise.all(batch.map((child) => walk(child, path, out)));
  }
}
