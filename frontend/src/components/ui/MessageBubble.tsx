// components/MessageBubble.tsx

import ReactMarkdown from "react-markdown";
interface MessageBubbleProps {
  content: string;
  role: "user" | "assistant";
}


export default function MessageBubble({content,role}: MessageBubbleProps) 
{

  return (

    <div
      className={
        role === "user"
          ? "message-content user-message"
          : "message-content assistant-message"
      }
    >

      <ReactMarkdown>
        {content}
      </ReactMarkdown>

    </div>
  );
}