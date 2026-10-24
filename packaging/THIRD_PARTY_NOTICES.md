# Third-party notices

The AppImage bundles these components in addition to the project's own code.

## Orthority

- License: GNU Affero General Public License v3.0 or later
  (AGPL-3.0-or-later).
- Upstream source: https://github.com/leftfield-geospatial/orthority
- Version: pinned in `pyproject.toml` and `uv.lock`.
- The bundled copy is unmodified. The full license text is shipped in
  `licenses/ORTHORITY-AGPL-3.0-or-later.txt`.

AGPL section 6 requires the Corresponding Source for a conveyed object-code
work. Orthority's source is available at the upstream URL above and in this
project's source distribution. If you modify Orthority, publish the modified
source as the AGPL requires.

## OpenCV

- License: Apache-2.0.
- Upstream source: https://github.com/opencv/opencv-python
- The build installs `opencv-python-headless` (not the GUI wheel) so the bundle
  does not require libGL or X11.
