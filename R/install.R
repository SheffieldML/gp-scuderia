# Install the R packages the R runners need (CIP-0007).
#   Rscript R/install.R ci/pinned-r.txt   # exact versions, "package version" per line
#   Rscript R/install.R ci/latest-r.txt   # newest CRAN release, "package" per line
# Packages go to the first entry of R_LIBS_USER, which must exist. Installed packages at the
# requested version are left alone. A pinned version that CRAN has since archived is fetched
# from the CRAN archive.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1)
lib <- strsplit(Sys.getenv("R_LIBS_USER"), .Platform$path.sep)[[1]][1]
if (is.na(lib) || !nzchar(lib)) stop("Set R_LIBS_USER to the library directory to install into")
dir.create(lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(lib, .libPaths()))
repo <- "https://cloud.r-project.org"

lines <- trimws(readLines(args[1]))
lines <- lines[nzchar(lines) & !startsWith(lines, "#")]
for (l in lines) {
  f <- strsplit(l, "[[:space:]]+")[[1]]
  pkg <- f[1]; ver <- if (length(f) > 1) f[2] else NA
  have <- tryCatch(as.character(packageVersion(pkg, lib.loc = .libPaths())), error = function(e) NA)
  if (!is.na(have) && (is.na(ver) || have == ver)) { cat(pkg, have, "present\n"); next }
  if (is.na(ver)) {
    install.packages(pkg, lib = lib, repos = repo)
  } else {
    current <- available.packages(repos = repo)[pkg, "Version"]
    url <- if (identical(unname(current), ver)) NULL else
      sprintf("%s/src/contrib/Archive/%s/%s_%s.tar.gz", repo, pkg, pkg, ver)
    if (is.null(url)) install.packages(pkg, lib = lib, repos = repo)
    else install.packages(url, lib = lib, repos = NULL, type = "source")
  }
  got <- tryCatch(as.character(packageVersion(pkg, lib.loc = lib)), error = function(e) NA)
  if (is.na(got) || (!is.na(ver) && got != ver)) stop("Could not install ", pkg, if (!is.na(ver)) paste0(" ", ver))
  cat(pkg, got, "installed\n")
}
