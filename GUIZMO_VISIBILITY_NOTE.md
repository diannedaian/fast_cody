# Guizmo Visibility Issue

## Problem
The guizmo (affine handle) remains visible when switching to keyboard control mode, despite setting `guizmo.visible = False`.

## Root Cause
The guizmo is initialized with `init_guizmo(True, ...)` which sets it up to be visible. ImGuizmo (the underlying library) may be rendering the widget in screen space regardless of the `visible` property, especially if it's actively being used or if the viewer's rendering system doesn't properly respect the visibility flag.

## Current Workaround
The code now:
1. Sets `guizmo.visible = False` every frame in keyboard mode
2. Tries to disable it through multiple properties (`enabled`, `active`, `set_visible`)
3. Avoids updating the guizmo transform in keyboard mode to prevent it from being "activated"

## Potential Solutions

### Option 1: Conditional Initialization (Requires C++ Changes)
Modify the viewer to support conditional guizmo initialization or provide a way to completely disable/enable it at runtime.

### Option 2: Screen-Space Rendering Check (Requires C++ Changes)
Modify ImGuizmo rendering to check the visibility property before rendering, or add a flag to completely skip rendering when disabled.

### Option 3: Accept the Limitation
If the guizmo is still visible but non-interactive in keyboard mode, this might be acceptable. The user can still use keyboard controls effectively.

## Testing
To verify if the workaround is working:
1. Switch to keyboard mode (K key)
2. Try to interact with the guizmo - it should not respond
3. The guizmo might still be visible but should be "dead" (non-interactive)

## Future Improvement
Consider modifying the C++ viewer code to add a proper `disable_guizmo()` / `enable_guizmo()` method that completely stops rendering and interaction.

