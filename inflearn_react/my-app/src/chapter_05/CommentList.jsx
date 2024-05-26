import React from "react";
import Comment from "./Comment";

const comments = [
  {
    name: "테스터",
    comment: "안녕하세요, 반가워요.",
  },
  {
    name: "음악가",
    comment: "음악 좋아요.",
  },
];

function CommentList(props) {
  return (
    <div>
      {comments.map((comment) => {
        return <Comment name={comment.name} comment={comment.comment} />;
      })}
    </div>
  );
}

export default CommentList;