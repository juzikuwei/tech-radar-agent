export function UserBubble({ question }: { question: string }) {
  return (
    <div className="user-row">
      <div className="user-bubble">{question}</div>
    </div>
  );
}
