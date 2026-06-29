// Feedback-based user rating. Ignoring feedback prompts lowers the rating,
// which can later inform routing/queue priority.
import { query } from "./db.js";

export async function notePrompted(userId: string | null): Promise<void> {
  if (!userId) return;
  await query(
    `update users set feedback_prompts = feedback_prompts + 1,
       rating = greatest(0.1, least(1.5,
         (feedback_given::float / (feedback_prompts + 1)) + 0.25))
     where id = $1`,
    [userId],
  );
}

export async function noteFeedback(userId: string | null): Promise<void> {
  if (!userId) return;
  await query(
    `update users set feedback_given = feedback_given + 1,
       rating = greatest(0.1, least(1.5,
         ((feedback_given + 1)::float / greatest(feedback_prompts, 1)) + 0.25))
     where id = $1`,
    [userId],
  );
}
