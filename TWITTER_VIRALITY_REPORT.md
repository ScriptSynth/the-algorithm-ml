# Twitter Algorithm Analysis Report: How to Go Viral on Twitter

## Executive Summary

This comprehensive report analyzes Twitter's open-source machine learning algorithm repository to understand exactly how tweets and accounts are ranked on the platform. Based on this analysis, we provide actionable recommendations for maximizing your Twitter account's reach and increasing the likelihood of going viral.

**Key Finding**: Twitter uses a sophisticated multi-stage ranking system that evaluates over 1000 features to predict user engagement and scores tweets accordingly. The "Heavy Ranker" model assigns weights to different engagement types, with reply engagement by the author being the most valuable (+75.0 weight) and reports being the most harmful (-369.0 weight).

---

## Table of Contents

1. [Understanding Twitter's Ranking System](#understanding-twitters-ranking-system)
2. [How Tweets Are Scored](#how-tweets-are-scored)
3. [Key Engagement Metrics and Their Weights](#key-engagement-metrics-and-their-weights)
4. [Features That Influence Tweet Ranking](#features-that-influence-tweet-ranking)
5. [How to Make Your Twitter Account Go Viral](#how-to-make-your-twitter-account-go-viral)
6. [Advanced Strategies](#advanced-strategies)
7. [What to Avoid](#what-to-avoid)

---

## 1. Understanding Twitter's Ranking System

### The Architecture

Twitter's "For You" timeline uses a multi-stage recommendation system:

1. **Candidate Retrieval**: Identifies potential tweets from various sources
2. **Light Ranking**: Quick filtering using simpler models (Earlybird)
3. **Heavy Ranking**: Sophisticated ML model that scores remaining tweets
4. **Filtering & Heuristics**: Final safety and quality checks

### The Heavy Ranker Model

The Heavy Ranker is a **MaskNet** neural network architecture that:
- Processes **hundreds of input features** about the tweet, author, and viewer
- Outputs **probabilities** for 10 different engagement types
- Combines these probabilities into a **final score** using weighted sum
- Uses **TwHIN embeddings** (200-dimensional vectors) to understand user-author and user-tweet relationships

---

## 2. How Tweets Are Scored

### The Scoring Formula

```
Final Score = Σ (Weight of Engagement Type i) × (Probability of Engagement i)
```

### The Scoring Process

For each tweet shown to a user, the model:

1. **Collects Features**: Gathers ~1000+ features including:
   - User historical behavior (50-day aggregates)
   - Real-time signals (30-minute aggregates)
   - Tweet characteristics
   - User-author relationship strength
   - Content type and quality signals

2. **Predicts Engagement**: Uses the neural network to predict the probability that the user will:
   - Like/favorite the tweet
   - Retweet it
   - Reply to it
   - Click into the conversation
   - Watch video content
   - Provide negative feedback
   - Report the tweet
   - And more...

3. **Calculates Score**: Multiplies each probability by its weight and sums them up

4. **Ranks Tweets**: Orders all candidate tweets by their final score

---

## 3. Key Engagement Metrics and Their Weights

Here are the exact weights Twitter uses (as of the algorithm's release):

| Engagement Type | Weight | What It Means |
|----------------|--------|---------------|
| **Reply Engaged by Author** | **+75.0** | You reply and the author engages with your reply |
| **Good Profile Click** | **+12.0** | User clicks your profile and likes/replies to a tweet |
| **Reply** | **+13.5** | User replies to your tweet |
| **Good Click** | **+11.0** | User clicks into conversation and replies or likes |
| **Good Click v2** | **+10.0** | User clicks into conversation and stays 2+ minutes |
| **Retweet** | **+1.0** | User retweets your tweet |
| **Like/Favorite** | **+0.5** | User likes your tweet |
| **Video Playback 50%** | **+0.005** | User watches 50%+ of video |
| **Negative Feedback** | **-74.0** | User clicks "show less often" or blocks/mutes |
| **Report** | **-369.0** | User reports your tweet |

### Critical Insights

1. **Replies are KING**: Getting replies is worth **27x more** than getting likes (based on weight ratios: 13.5 vs 0.5)
2. **Author Engagement**: When you reply to others and they engage back, it's the most valuable signal (+75.0 weight)
3. **Conversation Quality**: Clicks that lead to sustained engagement (2+ min) are extremely valuable (+10.0 to +12.0 weights)
4. **Negative Signals**: Reports are catastrophic (-369.0 weight) and negative feedback is very harmful (-74.0 weight)

---

## 4. Features That Influence Tweet Ranking

The model analyzes over **1000 features** grouped into several categories:

### 4.1 Aggregate Features (Historical Behavior)

These track your account's historical performance over two time windows:
- **Long-term**: 50-day rolling aggregates
- **Real-time**: 30-minute to 3-day aggregates

#### Author Aggregate Features
- How many likes, retweets, replies your tweets typically get
- How often people click on your profile
- Video completion rates on your content
- Negative feedback rates

#### User-Author Relationship Features
Tracks the specific relationship between the viewer and you:
- Have they engaged with your content before?
- How frequently do they interact with you?
- What types of content do they engage with from you?
- Time since last interaction

#### User Aggregate Features
Tracks what the viewer typically engages with:
- What type of content do they like?
- Who do they typically engage with?
- Their engagement patterns over time

### 4.2 Tweet Content Features

**Media Features**:
- `has_image`, `has_video`, `has_native_video`
- `video_duration`, `aspect_ratio`
- `has_multiple_media`
- Number of photos

**Text Features**:
- `text.length`
- `has_question` (questions in text)
- `num_hashtags`, `num_mentions`
- `has_link`, `link_count`
- Language matching with viewer

**Tweet Type**:
- `is_reply`, `is_retweet`
- `is_extended_reply` (threaded replies)
- `has_quote`
- Original vs retweet

**Content Quality Signals**:
- `from_verified_account`
- `is_author_bot`, `is_author_spam`, `is_author_new`
- `is_offensive`, `is_sensitive`
- Abuse/spam flags

### 4.3 Real-Time Engagement Signals

**Tweet-Level Metrics** (updated every 30 minutes):
- Current favorite count
- Current retweet count
- Current reply count
- Video view counts
- Negative feedback counts

**Decayed Counts**:
Twitter uses time-decayed engagement counts that give more weight to recent engagement:
- `decayed_favorite_count`
- `decayed_retweet_count`
- `decayed_reply_count`

### 4.4 Social Graph Features

**Two-Hop Features**:
If User A follows User B, and User B likes your tweets, this creates a positive "two-hop" signal for User A seeing your content.

Types tracked:
- `favorite.favorited_by`
- `following.followed_by`
- `mutual_follow`
- `mentioned_by`, `retweeted_by`

**RealGraph Features**:
Measures real relationship strength based on:
- `num_direct_messages`
- `num_favorites`, `num_retweets`
- `num_profile_views`, `num_link_clicks`
- `total_dwell_time`
- `num_address_book_email/phone` (if contacts are connected)
- Days since last interaction

### 4.5 TwHIN Embeddings

Three 200-dimensional embeddings capture similarity:

1. **User Follow Embeddings**: 
   - Who is likely to follow you
   - Who you are likely to follow

2. **User Engagement Embeddings**:
   - What content users typically engage with
   - Similarity between viewers and your content

These embeddings help Twitter understand users who may not directly follow you but have similar interests to those who engage with your content.

### 4.6 Topic and Interest Features

**Author-Topic Aggregates**:
- Performance of your content on specific topics
- User engagement with specific topics

**User-Topic Features**:
- Topics the user is interested in
- Topic-based negative feedback

### 4.7 Timing Features

**Tweet Age**:
- `time_since_tweet_creation`
- `tweet_age_ratio`
- Time between engagements

**User Timing**:
- `time_since_viewer_account_creation`
- Session context (start, middle, end)
- Polling vs organic requests

---

## 5. How to Make Your Twitter Account Go Viral

Based on the algorithm analysis, here's your roadmap to Twitter success:

### Strategy 1: Maximize Reply Engagement (Highest Value)

**Why**: Replies have a weight of 13.5, which is 27x higher than likes (0.5) and 13.5x higher than retweets (1.0).

**How**:
1. **Ask Questions**: The algorithm specifically tracks `has_question` as a feature
   - End tweets with thought-provoking questions
   - Use "What do you think?", "How would you solve this?", etc.
   - Create polls (engagement bait that works)

2. **Create Conversation Starters**:
   - Share controversial (but not offensive) opinions
   - Post incomplete thoughts that need input
   - Use "Hot take:" or "Unpopular opinion:" formats

3. **Reply to Your Own Replies** (75x multiplier!):
   - When someone replies to your tweet, ALWAYS engage back
   - This creates the highest-value signal: `reply_engaged_by_author`
   - Sets up a feedback loop where the algorithm learns your content generates author engagement

4. **Thread Your Content**:
   - Break content into threads
   - Each reply in a thread counts as engagement
   - Use "1/" numbering to set expectations

5. **Be Controversial (Carefully)**:
   - Hot takes generate replies
   - Stay away from content that triggers negative feedback or reports
   - Debate, don't hate

### Strategy 2: Build Strong User-Author Relationships

**Why**: The algorithm heavily weights historical interaction patterns between specific users.

**How**:
1. **Engage Consistently with Your Audience**:
   - Reply to people who comment on your tweets
   - Like their replies to show appreciation
   - Create a pattern of interaction that the algorithm recognizes

2. **Identify Your "Super Fans"**:
   - People who regularly engage with you get a strong relationship score
   - Prioritize engaging with them
   - Their future engagements will be weighted more heavily

3. **Build Two-Hop Relationships**:
   - Engage with accounts that your followers follow
   - Retweet and comment on their content
   - This creates indirect paths for your content to reach new audiences

4. **Use Direct Messages Strategically**:
   - `num_direct_messages` is a RealGraph feature
   - Building real relationships via DMs strengthens the algorithm's view of your connections

### Strategy 3: Optimize for Profile Clicks (+12x multiplier)

**Why**: Getting users to click your profile AND engage with other tweets is extremely valuable.

**How**:
1. **Pin Your Best Tweet**:
   - Make it highly engaging
   - Ensure it prompts action (reply, like, retweet)

2. **Curate Your Recent Tweets**:
   - Keep quality high
   - Delete or unpin weak performing content
   - Make your profile worth exploring

3. **Use Clear Branding**:
   - Professional header image
   - Clear bio explaining who you are
   - Consistency in content themes

4. **Tweet Consistently**:
   - Multiple quality tweets per day
   - When someone visits your profile, there should be fresh content

### Strategy 4: Optimize Content Type and Format

**Based on feature importance**:

1. **Use Native Media** (algorithm tracks this specifically):
   - `has_native_video` and `has_native_image` are distinct features
   - Upload directly to Twitter rather than linking
   - Multiple images are tracked: `has_multiple_media`

2. **Video Strategy**:
   - Create videos that people watch 50%+ (`video_playback_50` feature)
   - First 50% must be engaging enough to retain viewers
   - Optimal duration: Long enough to be substantial, short enough to watch fully
   - Use video for complex topics that benefit from demonstration

3. **Image Strategy**:
   - Tweets with images get different treatment
   - Use high-quality, attention-grabbing visuals
   - Infographics, charts, and data visualizations perform well

4. **Text Optimization**:
   - `text.length` is a feature - find the sweet spot
   - Too short: May appear low-effort
   - Too long: May reduce readability
   - Optimal: 100-280 characters for maximum engagement

5. **Link Strategy**:
   - Links are tracked: `has_link`, `link_count`, `has_visible_link`
   - Twitter may slightly deweight links (keeps users on platform)
   - When using links:
     - Add compelling context/commentary
     - Use link in addition to media, not instead of
     - Make the tweet valuable even without clicking

### Strategy 5: Timing and Consistency

**Real-time Aggregates Matter**:

1. **Post When Your Audience is Active**:
   - 30-minute aggregates mean early engagement snowballs
   - Initial engagement creates momentum
   - Find your audience's peak activity times

2. **Maintain Posting Consistency**:
   - `author_aggregate` features track your long-term performance
   - Consistent quality beats sporadic virality
   - Algorithm learns your typical engagement rates

3. **Build on Momentum**:
   - When a tweet performs well, post related content quickly
   - Ride the wave of increased visibility
   - Engage heavily with replies during peak performance

### Strategy 6: Network Effects and Virality Mechanics

**Understanding Distribution**:

1. **Get Early High-Value Engagement**:
   - First 30 minutes are critical (real-time aggregates)
   - One reply from a high-engagement user > 10 likes from low-engagement users
   - Seed your content with your most engaged followers first

2. **Leverage Your Network**:
   - Accounts that regularly engage with you will see your content more
   - Their engagements count more heavily
   - Build a core group of engaged followers

3. **Cross-Network Amplification**:
   - Retweets expose your content to new networks
   - Each new network has its own engagement potential
   - Create "retweet-worthy" content (surprising, useful, funny)

4. **Topic Clustering**:
   - `author-topic_aggregate` features track your performance by topic
   - Build authority in specific topics
   - Topic consistency helps algorithm understand your niche
   - Occasional topic variation can expose you to new audiences

### Strategy 7: Avoid Negative Signals

**Critical**: Negative signals are weighted heavily against you.

1. **Never Get Reported** (-369x penalty):
   - Avoid controversial statements about protected groups
   - No spam, no manipulation tactics
   - Don't use banned engagement tactics (follow/unfollow, automation)
   - Respect community guidelines absolutely

2. **Minimize Negative Feedback** (-74x penalty):
   - Track your follower quality, not just quantity
   - Unengaged followers may use "show less often"
   - Post content that resonates with YOUR audience, not everyone
   - Don't post too frequently (can trigger "show less often")

3. **Avoid Spam Flags**:
   - Features: `is_author_spam`, `label_spam_flag`, `label_spam_hi_rcl_flag`
   - Don't overuse hashtags (`num_hashtags` is tracked)
   - Don't over-mention users
   - Avoid repetitive content
   - Space out promotional content

4. **Quality Over Quantity**:
   - `tweet_count_from_user_in_snapshot` is tracked
   - Too many tweets can dilute your engagement rate
   - Better to tweet 2-3 quality posts than 10 mediocre ones

---

## 6. Advanced Strategies

### Strategy A: Engagement Arbitrage

**Concept**: Different engagement types have different values but similar psychological triggers.

**Execution**:
1. Design content specifically for replies, not likes
2. Use formats that require text responses (questions, debates, fill-in-the-blanks)
3. Make content that's easy to reply to but requires thought to like

**Example**:
- ❌ "AI is amazing!" (generates likes)
- ✅ "What's your biggest AI concern?" (generates replies)

### Strategy B: The Reply Loop Strategy

**Concept**: Maximize the 75x "reply engaged by author" multiplier.

**Execution**:
1. Post content that generates thoughtful replies
2. Reply substantively to EVERY reply in first hour
3. Ask follow-up questions in your replies
4. Create mini-conversations in your replies
5. This signals to the algorithm that your tweets generate high-quality discussions

**Expected Outcome**:
- Builds strong user-author relationship scores
- Increases distribution of future tweets
- Creates loyal community

### Strategy C: Profile Click Funnel

**Concept**: Optimize for the +12x profile click multiplier.

**Execution**:
1. Use provocative or intriguing tweet hooks
2. Reference "see pinned tweet" or "more in my profile"
3. Have immediately engaging content on profile
4. Use bio to encourage specific actions
5. Ensure recent tweets prompt engagement

**Example**:
Tweet: "After 5 years of research, I've identified the 3 factors that predict Twitter virality. Thread: 🧵 (or see pinned)"

### Strategy D: The Video Engagement Hack

**Concept**: Video has special treatment but low weight (+0.005 for 50% playback).

**Insight**: Use video to generate OTHER high-value engagements.

**Execution**:
1. Create videos that prompt discussion (not just passive watching)
2. End videos with questions
3. Use video to explain complex topics that generate replies
4. Include text in video that's screenshot-worthy
5. Make the tweet text highly engaging even without watching video

**Outcome**: Video is the wrapper, replies are the goal.

### Strategy E: Topic Authority Building

**Concept**: Build strong `author-topic_aggregate` scores.

**Execution**:
1. Choose 2-3 core topics
2. Post consistently on these topics for 50+ days
3. Track which subtopics generate most engagement
4. Double down on high-performing subtopics
5. Occasionally branch to related topics to expand reach

**Timeline**: 
- Days 1-50: Build aggregate data
- Days 50+: Algorithm has strong signal of your topic authority
- Months 3+: You're recommended to users interested in your topics

### Strategy F: The Verified Account Advantage

**Concept**: `from_verified_account` is a tracked feature.

**Reality**:
- Twitter Blue/Premium subscription gets you verification
- Verification is a signal of account legitimacy
- May provide small boost in distribution
- More important: Reduces spam/bot penalties

**ROI**: If you're serious about Twitter growth, verification is likely worth it.

### Strategy G: Network Seeding

**Concept**: Strategically use early engagement to trigger distribution.

**Execution**:
1. Post your content
2. Immediately share with 3-5 highly engaged followers via DM
3. Ask for genuine engagement (not "please RT" - that's against TOS)
4. The quality early engagement triggers real-time aggregates
5. This can create snowball effect

**Warning**: This must be organic. Don't create engagement pods - that's manipulative and can get you banned.

### Strategy H: Multi-Modal Content Strategy

**Concept**: Different content types serve different purposes.

**Framework**:

| Content Type | Primary Goal | Frequency |
|--------------|-------------|-----------|
| Question Tweets | Maximize replies | Daily |
| Image Posts | Broad reach, likes | 2-3x/week |
| Video Content | Deep engagement | Weekly |
| Threads | Profile clicks | 2x/week |
| Quick Takes | Consistency | Fill gaps |

**Execution**:
- Rotate content types
- Track performance by type
- Optimize within each category

---

## 7. What to Avoid

Based on the algorithm's negative signals and penalties:

### ❌ Don't: Engagement Bait (Against TOS)
- "RT if you agree"
- "Follow me for more"
- Fake contests
- **Why**: Can trigger spam flags and reports (-369x)

### ❌ Don't: Overuse Hashtags
- Feature: `num_hashtags` is tracked
- More than 2-3 hashtags appears spammy
- `has_multiple_hashtag_or_trend` is specifically tracked
- **Impact**: May trigger spam classification

### ❌ Don't: Over-mention Users
- Feature: `num_mentions` is tracked
- Looks like spam or clout-chasing
- **Impact**: Reduces credibility

### ❌ Don't: Post Too Frequently
- Feature: `tweet_count_from_user_in_snapshot`
- Reduces average engagement per tweet
- Can trigger "show less often"
- **Sweet Spot**: 3-10 quality tweets per day

### ❌ Don't: Use External Links Exclusively
- Features track links: `has_link`, `link_count`
- Twitter prefers to keep users on platform
- **Better**: Use links with compelling native content

### ❌ Don't: Ignore Replies
- Missing the 75x multiplier
- Weakens user-author relationship scores
- **Impact**: Future tweets get less distribution to those users

### ❌ Don't: Post Low-Quality Content
- Every tweet affects your aggregate scores
- Poor performing tweets drag down your averages
- **Strategy**: Delete genuinely bad tweets

### ❌ Don't: Be Offensive
- Features: `is_offensive`, `is_sensitive`
- Triggers negative feedback and reports
- **Impact**: -74x to -369x penalties

### ❌ Don't: Buy Followers
- Creates poor engagement ratios
- Low-quality followers don't engage
- Damages aggregate metrics
- **Risk**: Account suspension

### ❌ Don't: Use Automation Irresponsibly
- Against TOS if it appears bot-like
- Features: `is_author_bot`
- **Risk**: Spam classification or ban

### ❌ Don't: Post Only Promotional Content
- Generates negative feedback
- Poor engagement ratios
- **Ratio**: 80% value, 20% promotion (at most)

---

## Conclusion: The Virality Formula

Based on Twitter's algorithm, here's the formula for going viral:

### The Perfect Tweet
1. **Content**: Asks a thought-provoking question or shares a surprising insight
2. **Format**: Native image or video + compelling text
3. **Length**: 100-250 characters for optimal engagement
4. **Timing**: Posted when your audience is most active
5. **Follow-up**: You actively reply to every comment in the first hour
6. **Topic**: Within your established authority areas

### The Perfect Account
1. **Consistency**: 3-5 high-quality tweets per day
2. **Engagement**: Replies to all comments on your tweets
3. **Relationships**: Strong connections with core followers
4. **Content Mix**: 70% discussions (replies), 20% media, 10% threads
5. **Quality**: High engagement rates, minimal negative feedback
6. **Authority**: Known for 2-3 specific topics
7. **Verification**: Verified account with complete profile

### The Growth Timeline

**Month 1: Foundation**
- Build consistent posting schedule
- Identify core topics
- Begin engaging heavily with community
- Goal: Establish baseline aggregate metrics

**Month 2: Optimization**
- Test different content formats
- Identify what generates replies
- Build user-author relationships
- Goal: Improve engagement rates

**Month 3: Acceleration**
- Double down on what works
- Increase volume slightly
- Leverage network effects
- Goal: First viral tweet (1000+ engagements)

**Months 4-6: Scaling**
- Maintain consistency
- Build topic authority
- Expand into related topics
- Goal: Regular high-performing tweets

**Months 6+: Virality**
- Strong aggregate metrics across 50 days
- Algorithm recognizes your content quality
- New users regularly exposed to your content
- Goal: Multiple viral tweets, sustainable growth

### The Ultimate Insight

**Twitter's algorithm rewards one thing above all: meaningful conversation.**

The highest-weighted signals are all about:
- Replies (especially author engagement)
- Profile clicks with sustained engagement
- Conversation depth and duration

If you optimize for genuine, valuable discussions rather than vanity metrics like likes and retweets, the algorithm will reward you with increased distribution.

**Your goal isn't to game the algorithm—it's to create the kind of content the algorithm is designed to promote: content that generates real human connection and conversation.**

---

## Final Recommendations

### For Immediate Impact (Next 7 Days)
1. Reply to every comment on your tweets
2. End every tweet with a question
3. Post at consistent times daily
4. Use native images/videos
5. Engage with your network's content

### For Sustainable Growth (Next 90 Days)
1. Choose 2-3 core topics and build authority
2. Post 3-5 quality tweets per day
3. Track which content types generate replies
4. Build relationships with engaged followers
5. Monitor and minimize negative feedback
6. Maintain high engagement rate over likes/followers

### For Maximum Virality (Long-term)
1. Become known for specific valuable insights
2. Build a community that actively discusses your content
3. Create content specifically designed for replies
4. Maintain consistency for 50+ days to build strong aggregates
5. Leverage network effects through two-hop connections
6. Continuously optimize based on performance data

---

**Remember**: The algorithm is sophisticated, but it's ultimately designed to serve users valuable content. Create genuinely valuable, conversation-worthy content, and the algorithm will amplify it.

Good luck going viral! 🚀
