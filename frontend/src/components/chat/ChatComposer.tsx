import { FormEvent, useEffect, useRef } from "react";


interface ChatComposerProps {
  draft: string;
  pending: boolean;
  onDraftChange: (value: string) => void;
  onSubmit: (value: string) => void;
}


export function ChatComposer({
  draft,
  pending,
  onDraftChange,
  onSubmit,
}: ChatComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const wasPending = useRef(pending);

  useEffect(() => {
    if (wasPending.current && !pending) {
      textareaRef.current?.focus();
    }
    wasPending.current = pending;
  }, [pending]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    // 自动增高：按内容 scrollHeight 设置高度，CSS max-height 负责封顶。
    textarea.style.height = "auto";
    if (textarea.scrollHeight > 0) {
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  }, [draft]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (pending) {
      return;
    }
    onSubmit(draft);
  }

  return (
    <form className="composer" onSubmit={handleSubmit}>
      <div className="composer-inner">
        <label className="visually-hidden" htmlFor="question">
          继续追问或开始一个新技术话题
        </label>
        <div className="composer-row">
          <textarea
            id="question"
            ref={textareaRef}
            value={draft}
            onChange={(event) => onDraftChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.nativeEvent.isComposing) {
                return;
              }
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                if (!pending) {
                  onSubmit(draft);
                }
              }
            }}
            placeholder="给研究助手发送消息"
            rows={1}
          />
          <button
            className="primary-button"
            type="submit"
            aria-label={pending ? "处理中" : "发送"}
            disabled={!draft.trim() || pending}
          >
            <span aria-hidden="true">{pending ? "…" : "↑"}</span>
          </button>
        </div>
        <div className="composer-footer">
          <span><i className="composer-status-dot" /> 本地知识库</span>
          <span>Enter 发送 · Shift + Enter 换行</span>
        </div>
      </div>
    </form>
  );
}
