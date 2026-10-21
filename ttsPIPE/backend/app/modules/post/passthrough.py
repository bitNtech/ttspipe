"""post/passthrough: identity."""
from app.core.base import ModuleInfo, PostModule
from app.core.registry import register


@register
class PassthroughPost(PostModule):
    info = ModuleInfo(id="post/passthrough", kind="post", name="Passthrough", description="Frames pass unchanged.",
                      builtin=True, tags=["builtin"])
